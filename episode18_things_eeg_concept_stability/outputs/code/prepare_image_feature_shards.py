#!/usr/bin/env python3
"""Trusted builder for all frozen EP18 images; no feature fitting or EEG.

The source ZIP, private credential and exact source crosswalk stay in this
process. Restricted temporary packs contain only metadata-free RGB arrays and
the original opaque IDs. No cleanup or extraction successor is automatic.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import zipfile

from image_pixel_worker import PilotContractError, bounded_process, require, write_json
from run_image_feature_pilot import JOIN, ROUTING, exact_members, private_password, qualified_assets, select_images
from run_image_feature_shard import IMAGE_COUNT, admit, load_config, planned_tasks, validate_manifest


def build_packs(archive_path, selected_members, cache, config, check, password_file=None):
    """Open source archive once and decode every assigned stored raster once."""
    import numpy as np
    from PIL import Image
    pack_size = config["pixel_pack_images"]
    packs, total_bytes, total_pixels, encrypted = [], 0, 0, 0
    with zipfile.ZipFile(archive_path) as source:
        wanted = {member for _, member in selected_members}
        entries = {}
        for entry in source.infolist():
            if entry.filename in wanted:
                require(entry.filename not in entries and not entry.is_dir(), "Ambiguous selected archive member")
                entries[entry.filename] = entry
        require(set(entries) == wanted, "A frozen source image is missing")
        encrypted = sum(bool(entries[member].flag_bits & 1) for _, member in selected_members)
        password = private_password(password_file) if encrypted else None
        for first in range(0, len(selected_members), pack_size):
            stop = min(first+pack_size, len(selected_members))
            name = f"pixels_{len(packs):05d}.npz"
            path = cache/name
            used = 0
            with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as pack:
                for key, member in selected_members[first:stop]:
                    check()
                    with source.open(entries[member], pwd=password) as stream, Image.open(stream) as image:
                        width, height = image.size
                        size = width*height*3
                        overhead = 1024 + (stop-first)*128
                        require(size > 0 and used+size+overhead <= config["pixel_pack_byte_cap"],
                                "Pixel pack exceeds selected allowance; no resizing or image exclusion")
                        require(total_bytes+used+size+overhead+config["metadata_byte_cap"]
                                <= config["pixel_cache_byte_cap"], "Total pixel cache allowance exceeded")
                        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                    require(pixels.shape == (height, width, 3), "Stored RGB raster dimensions changed")
                    with pack.open(key+".npy", "w", force_zip64=True) as destination:
                        np.lib.format.write_array(destination, pixels, allow_pickle=False)
                    total_pixels += pixels.nbytes
                    del pixels
                    used = pack.fp.tell()
                    check()
            actual = path.stat().st_size
            require(actual <= config["pixel_pack_byte_cap"], "Closed pixel pack exceeds selected allowance")
            total_bytes += actual
            require(total_bytes+config["metadata_byte_cap"] <= config["pixel_cache_byte_cap"],
                    "Closed pixel cache exceeds selected allowance")
            packs.append({"name": name, "start": first, "stop": stop, "bytes": actual})
            print(json.dumps({"closed_pixel_pack": len(packs)-1, "completed_images": stop,
                              "total_pixel_pack_bytes": total_bytes}), flush=True)
    return packs, {"decoded_RGB_bytes": total_pixels, "pixel_pack_bytes": total_bytes,
                   "encrypted_images": encrypted, "private_credential_notice_read_by_builder": bool(encrypted)}


def prepare(config, archive, password_file, check, on_create=None):
    cache, run = Path(config["pixel_cache_dir"]), Path(config["run_dir"])
    require(not cache.exists() and not run.exists(), "Preserve prior cache and full-extraction attempts")
    cache.mkdir(mode=0o700)
    run.mkdir(mode=0o700)
    if on_create is not None:
        on_create(run)
    (run/"tasks").mkdir(mode=0o700)
    started = time.monotonic()
    assets = qualified_assets(sys.executable)
    with ROUTING.open(newline="", encoding="utf-8") as source:
        rows = select_images(csv.DictReader(source, delimiter="\t"), count=IMAGE_COUNT)
    require(all(row["main"] in {"True", "False"} and row["repeated"] in {"True", "False"} for row in rows),
            "Frozen image-role values changed")
    require(sum(row["main"] == "True" for row in rows) == 22248
            and sum(row["repeated"] == "True" for row in rows) == 200
            and all((row["main"] == "True") != (row["repeated"] == "True") for row in rows),
            "Full main/repeated image inventory changed")
    ids = [row["opaque_image_id"] for row in rows]
    with gzip.open(JOIN, "rt", newline="", encoding="utf-8") as source:
        selected = exact_members(csv.DictReader(source, delimiter="\t"), rows)
    del rows
    packs, delivery = build_packs(archive, selected, cache, config, check, password_file)
    del selected
    check()
    manifest = {"status": "complete_pixel_delivery", "image_count": IMAGE_COUNT, "opaque_ids": ids,
                "packs": packs, "assets": assets, "tasks": planned_tasks(config),
                "separation": "procedural_same_account_not_OS_isolation",
                "contents": "Original opaque IDs and stored-orientation RGB pixels; no source metadata",
                "temporary_research_image_cache": True,
                "cleanup": "Retain until all full features are verified; coordinator records a separate cleanup decision. No automatic deletion."}
    validate_manifest(manifest, config)
    write_json(cache/"manifest.json", manifest, config["metadata_byte_cap"])
    report = {"status": "full_image_pixel_delivery_completed", "analysis_id": "feature_preparation",
              "images": IMAGE_COUNT, "main_images": 22248, "repeated_images": 200, "packs": len(packs),
              "planned_family_tasks": len(manifest["tasks"]), **delivery,
              "elapsed_seconds": time.monotonic()-started,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
              "EEG_or_audit_event_access": False, "automatic_successor": False,
              "separation": "procedural_same_account_not_OS_isolation",
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run/"pixel_delivery_report.json", report, config["metadata_byte_cap"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument("--password-file", type=Path)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    owned_runs = []
    try:
        config = load_config(args.config)
        admit(config, "builder", args.deadline_unix)
        with bounded_process(args.deadline_unix, config["rss_bytes"]) as check:
            report = prepare(config, args.image_archive, args.password_file, check, on_create=owned_runs.append)
        print(json.dumps(report), flush=True)
        return 0
    except Exception as error:
        failure = {"status": "full_image_pixel_delivery_failure", "error_type": type(error).__name__}
        if isinstance(error, PilotContractError):
            failure["contract_diagnostic"] = str(error)
        if owned_runs:
            run = owned_runs[0]
            write_json(run/"pixel_delivery_failure.json", failure, 65536)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
