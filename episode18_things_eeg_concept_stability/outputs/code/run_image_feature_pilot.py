#!/usr/bin/env python3
"""Trusted builder for the fixed first-32 opaque-ID EP18 image cost pilot.

Only this process reads the frozen private routing and development event join.
Fresh workers receive RGB arrays, existing opaque UUIDs and local model assets.
No EEG, audit events, network transfer, full-extraction release or successor.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import resource
import re
import socket
import subprocess
import sys
import time
import zipfile

import qualify_model_runtime as qualification
from feature_extractors import validate_ids
from image_pixel_worker import (BUNDLE_BYTES, DURABLE_BYTES, FAMILIES, PILOT_COUNT, PilotContractError,
                                bounded_process, require, write_json)
from qualify_structural_inputs import image_key

EPISODE = Path(__file__).resolve().parents[2]
ROUTING = EPISODE / "outputs/runs/feature_preparation/20261003-mapping01/private_source_routing.tsv"
JOIN = EPISODE / "outputs/runs/input_role_preparation/20261003-freeze_metadata01/development_event_image_join.tsv.gz"
MAPPING = EPISODE / "outputs/config/model_runtime_asset_map_proposal.json"
QUALIFICATION = EPISODE / "outputs/runs/feature_preparation/exact_cpu_provision_46842156/model_runtime/model_runtime_qualification.json"
MAX_SECONDS = 6900


class PasswordNoticeError(ValueError):
    """Private archive notice has no single explicit, supported password field."""


def private_password(notice_path):
    """Read authorized source notice only in builder; never expose its contents.

    Accept exactly one explicitly password-labelled field, including bounded
    descriptive text before its colon, or a document consisting only of one
    ASCII token. Other prose formats stop without guessing words.
    """
    if notice_path is None:
        raise PasswordNoticeError()
    with notice_path.open("rb") as source:
        value = source.read(65537)
    if len(value) > 65536:
        raise PasswordNoticeError()
    text = value.decode("utf-8-sig")
    fields = re.findall(r"(?im)\bpassword[^\r\n:]{0,120}:[ \t]*([^\r\n]*)", text)
    fields += re.findall(r"(?im)^[ \t]*(?:password|zip password|archive password)[ \t]*=[ \t]*([^\r\n]*)", text)
    if fields:
        if len(fields) != 1:
            raise PasswordNoticeError()
        candidate = fields[0].strip()
    else:
        candidate = text.strip()
    if re.fullmatch(r"[!-~]{1,256}", candidate) is None:
        raise PasswordNoticeError()
    return candidate.encode("ascii")


def select_images(rows, count=PILOT_COUNT, expected_total=22448):
    rows = list(rows)
    require(len(rows) == expected_total, "Frozen image routing inventory differs")
    ids = [row["opaque_image_id"] for row in rows]
    validate_ids(ids, len(rows))
    canonical = [row["canonical_image"] for row in rows]
    require(len(set(canonical)) == len(rows) and all(canonical), "Frozen source identities are not unique")
    require(0 < count <= len(rows), "Invalid fixed pilot size")
    return sorted(rows, key=lambda row: row["opaque_image_id"])[:count]


def exact_members(rows, selected, expected_events=24648, expected_unique=22448):
    """Use the first already-frozen development person; never inspect audit view."""
    mapping, person, events = {}, None, 0
    for row in rows:
        if person is None:
            person = row["participant"]
        if row["participant"] != person:
            break
        events += 1
        canonical, member = row["canonical_image"], row["archive_member"]
        require(image_key(member) == canonical, "Frozen archive join no longer matches source identity")
        require(canonical not in mapping or mapping[canonical] == member,
                "Frozen identity has conflicting archive members")
        mapping[canonical] = member
    require(events == expected_events and len(mapping) == expected_unique,
            "First development person's frozen image inventory is incomplete")
    require(all(row["canonical_image"] in mapping for row in selected), "Selected frozen identity is absent")
    return [(row["opaque_image_id"], mapping[row["canonical_image"]]) for row in selected]


def build_pixel_pack(image_archive, selected_members, destination, check, byte_cap=BUNDLE_BYTES,
                     password_file=None):
    """Decode exact joined members; preserve stored raster orientation, strip metadata."""
    import numpy as np
    from PIL import Image

    ids = [key for key, _ in selected_members]
    validate_ids(ids, len(ids))
    require(ids == sorted(ids), "Pilot IDs must retain fixed order")
    used, raw_bytes = 0, 0
    with zipfile.ZipFile(image_archive) as source, zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as pack:
        # Reuse the frozen exact member, rejecting duplicate directory entries.
        wanted = {member for _, member in selected_members}
        members = {}
        for entry in source.infolist():
            if entry.filename in wanted:
                require(entry.filename not in members and not entry.is_dir(), "Ambiguous selected archive member")
                members[entry.filename] = entry
        require(set(members) == wanted, "Selected source image is missing")
        encrypted = sum(bool(members[member].flag_bits & 1) for _, member in selected_members)
        password = private_password(password_file) if encrypted else None
        for key, member in selected_members:
            check()
            with source.open(members[member], pwd=password) as stream, Image.open(stream) as original:
                width, height = original.size
                size = width * height * 3
                # Account for NPY and ZIP headers before allocating/writing raster.
                require(size > 0 and used + size + 1024 <= byte_cap,
                        "Decoded pixel pack allowance exceeded; no resizing or exclusion")
                array = np.asarray(original.convert("RGB"), dtype=np.uint8).copy()
            require(array.shape == (height, width, 3), "Decoded RGB shape changed")
            with pack.open(key + ".npy", "w", force_zip64=True) as target:
                np.lib.format.write_array(target, array, allow_pickle=False)
            raw_bytes += array.nbytes
            used = pack.fp.tell() + 128 * len(selected_members)
            del array
            require(used <= byte_cap, "Pixel pack byte allowance exceeded")
            check()
    actual = destination.stat().st_size
    require(actual <= byte_cap, "Closed pixel pack byte allowance exceeded")
    return {"images": len(ids), "decoded_RGB_bytes": raw_bytes, "pixel_pack_bytes": actual,
            "encrypted_selected_images": encrypted, "private_credential_notice_read_by_builder": bool(encrypted),
            "decode": "Pillow RGB; stored orientation; no EXIF transpose or ICC conversion; metadata stripped"}


def qualified_assets(executable):
    """Reuse the completed exact-runtime/source qualification without rehashing."""
    prior = qualification.read_json(QUALIFICATION)
    mapping = qualification.read_json(MAPPING)
    registry = qualification.read_json(qualification.REGISTRY)
    require(prior.get("status") == "asset_runtime_qualification_completed"
            and prior.get("asset_runtime_qualified") is True,
            "Completed asset/runtime qualification is required")
    require(prior["runtime"]["python"]["path"] == executable, "Use the qualified exact CPU interpreter")
    require(set(prior["load_results"]) == set(qualification.ROUTES)
            and set(prior["files"]) == set(qualification.ASSETS), "Incomplete retained qualification")
    require(registry["expanded_image_panel"] == list(FAMILIES), "Frozen six-family panel changed")
    for name, version in registry["runtime_recipe"]["target_versions"].items():
        require(prior["runtime"][name]["version"].split("+")[0] == version, "Qualified runtime version changed")
    require(mapping["schema"] == "ep18_model_runtime_asset_map_v1"
            and mapping["asset_root"] == "outputs/cache/model_assets", "Unselected asset map")
    root = EPISODE / mapping["asset_root"]
    assets = {}
    for name, (model_id, basename) in qualification.ASSETS.items():
        entry = mapping["assets"][name]
        path = root / entry["local_path"]
        require(path.name == basename and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                "Qualified asset identity changed")
        if model_id:
            require(entry["model_id"] == model_id and entry["revision"] == basename, "Model revision changed")
        evidence = {item["path"]: item for item in prior["files"][name]}
        paths = [path/item["name"] if model_id else path for item in entry["files"]]
        require(set(evidence) == {str(value) for value in paths}, "Retained qualified file identities differ")
        for value, item in zip(paths, entry["files"]):
            old = evidence[str(value)]
            require(old["source_integrity_checked"] is True and old["bytes"] == item["bytes"]
                    and old["source_url"] == item["url"] and value.stat().st_size == old["bytes"],
                    "Previously qualified asset changed")
        assets[name] = str(path)
    return assets


def worker_environment(local_dir):
    # Whitelist runtime/scheduler fields; never forward source metadata, tokens,
    # routing variables or an inherited PYTHONPATH into an image worker.
    keep = ("PATH", "LD_LIBRARY_PATH", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            "LANG", "LC_ALL")
    result = {key: os.environ[key] for key in keep if key in os.environ}
    result.update({key: "1" for key in qualification.THREADS})
    result.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1",
                  PYTHONDONTWRITEBYTECODE="1", HF_HOME=str(local_dir/"hf"),
                  XDG_CACHE_HOME=str(local_dir/"cache"), TMPDIR=str(local_dir))
    return result


def run(args):
    qualification.require_compute()
    require(os.environ.get("SLURM_MEM_PER_NODE") == "7168" and not os.environ.get("SLURM_JOB_GPUS"),
            "Selected pilot requires 1 CPU, 7 GiB and no GPU")
    require(math.isfinite(args.deadline_unix) and 0 < args.deadline_unix-time.time() <= MAX_SECONDS,
            "Use the selected positive shared allocation deadline, at most 6900 seconds")
    output, local = args.output_dir.absolute(), args.local_dir.absolute()
    require(output.parent.resolve() == (EPISODE/"outputs/runs/feature_preparation").resolve()
            and output.name.startswith("image_feature_pilot_") and not output.exists(), "Use a fresh episode pilot attempt")
    require(os.environ.get("L_SCRATCH_JOB") and local.parent.resolve() == Path(os.environ["L_SCRATCH_JOB"]).resolve()
            and not local.exists(), "Use a fresh direct child of this allocation's local scratch")
    output.mkdir()
    local.mkdir()
    started = time.monotonic()
    report = {"analysis_id": "feature_preparation", "phase": "first32_image_cost_pilot", "status": "running",
              "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "deadline_unix": args.deadline_unix,
              "selection": "first32_existing_opaque_UUIDs_lexicographic", "batch_size": 1,
              "inferences_per_family": 33, "repeatability": "first_image_once_more_at_batch1_atol1e-6_rtol1e-5_exact_details",
              "selected_images": PILOT_COUNT, "completed_families": [], "families": {},
              "separation": "procedural_same_account_not_OS_isolation", "audit_seal": False,
              "EEG_or_audit_event_access": False, "full_extraction_released": False, "automatic_successor": False,
              "qualification_reused": str(QUALIFICATION), "local_pixel_pack_cap_bytes": BUNDLE_BYTES,
              "durable_cap_bytes": DURABLE_BYTES}
    try:
        with bounded_process(args.deadline_unix) as check:
            assets = qualified_assets(sys.executable)
            with ROUTING.open(newline="", encoding="utf-8") as source:
                selected = select_images(csv.DictReader(source, delimiter="\t"))
            with gzip.open(JOIN, "rt", newline="", encoding="utf-8") as source:
                selected_members = exact_members(csv.DictReader(source, delimiter="\t"), selected)
            before = time.monotonic()
            bundle = local/"pixels.npz"
            report["pixel_delivery"] = build_pixel_pack(args.image_archive, selected_members, bundle, check,
                                                        password_file=args.password_file)
            report["pixel_delivery"]["seconds"] = time.monotonic()-before
            # Crosswalk remains in the trusted builder only; no durable copy.
            del selected_members
            ids = [row["opaque_image_id"] for row in selected]
            del selected
            write_json(local/"asset_paths.json", assets, 65536)
            write_json(output/"selected_ids.json", ids, 65536)
            env = worker_environment(local)
            for family in FAMILIES:
                check()
                command = [sys.executable, str(Path(__file__).with_name("image_pixel_worker.py")),
                           "--family", family, "--pixel-bundle", str(bundle), "--asset-paths", str(local/"asset_paths.json"),
                           "--output-dir", str(output/family), "--deadline-unix", str(args.deadline_unix)]
                # No stdout/stderr semantic/library diagnostics can escape to a
                # terminal. Worker report contains only selected operational fields.
                child = subprocess.run(command, cwd=local, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       timeout=max(.01, args.deadline_unix-time.time()))
                if child.returncode != 0:
                    report["failed_family"] = family
                    report["worker_returncode"] = child.returncode
                    raise RuntimeError("Fresh image worker failed; preserve partial attempt")
                result = qualification.read_json(output/family/"report.json")
                require(result["status"] == "complete_image_family" and result["family"] == family
                        and result["images"] == PILOT_COUNT and result["opaque_ids"] == ids,
                        "Family completion or opaque-ID coverage differs")
                report["completed_families"].append(family)
                report["families"][family] = result
                print(json.dumps({"completed_image_family": family, "images": PILOT_COUNT,
                                  "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
            check()
            report["status"] = "first32_image_cost_pilot_completed"
            report["projected_full22448_encoding_seconds"] = {
                family: sum(value["encoding_seconds"])*22448/PILOT_COUNT for family, value in report["families"].items()}
            report["projection_limit"] = "First32 UUID-sorted images only; encoding extrapolation excludes model load and I/O; no full extraction release."
    except Exception as error:
        report["status"] = "technical_image_cost_pilot_failure"
        report["error_type"] = type(error).__name__
        if isinstance(error, PilotContractError):
            report["contract_diagnostic"] = str(error)
    report.update(elapsed_seconds=time.monotonic()-started,
                  builder_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    # Only declared output files, no recursive filesystem scan or scientific read.
    paths = [output/"selected_ids.json"] + [output/family/name for family in FAMILIES
            for name in ("features.npy", "image_details.json", "report.json", "failure.json")]
    report["artifact_bytes_before_report"] = sum(path.stat().st_size for path in paths if path.exists())
    remaining = DURABLE_BYTES-report["artifact_bytes_before_report"]
    write_json(output/"image_feature_pilot_report.json", report, remaining)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument("--password-file", type=Path, help="Authorized private notice; trusted builder only, if ZIP encrypted")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        print(json.dumps({"status": "image_pilot_admission_failure", "error_type": type(error).__name__}), flush=True)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "completed_families", "elapsed_seconds")}), flush=True)
    return 0 if report["status"] == "first32_image_cost_pilot_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
