#!/usr/bin/env python3
"""Four fixed source-mask header witnesses; no structural voxel decoding."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib

import qualify_anatomical_axis as anatomy

nib, np = anatomy.nib, anatomy.np
Refusal, require = anatomy.Refusal, anatomy.require
SUBJECTS = anatomy.SUBJECTS
OUTPUT = anatomy.OUTPUT_PARENT / "units001"
INTERNAL_SECONDS = 120
REQUEST_SECONDS = 15
OBJECT_CAP = 8192
TOTAL_CAP = 28069
HEADER_BYTES = 352
FMRIPREP_COMMIT = "5155ba0d43836a4451141b90b5a70507f41205cf"
DATASET_ID = "81d8c126-b10f-4e8b-9eab-6825964c8552"
BASE = ("https://sftp.conp.ca/users/cneuromod/ria-conp/81d/"
        "8c126-b10f-4e8b-9eab-6825964c8552/annex/objects/")
WITNESSES = dict(zip(SUBJECTS, (
    {"directory": "XZ/5k", "bytes": 7602, "md5": "aaf135ebac3bced281281b703b20fe9f",
     "pointer_blob": "c58b7e29bc132eacac90a2074376b2aff81c3b41"},
    {"directory": "95/14", "bytes": 7127, "md5": "2510838dea8c0347033e769384730d38",
     "pointer_blob": "872a9a32863a5c3538c47e173a04f88cd77864c9"},
    {"directory": "Wg/7x", "bytes": 6862, "md5": "f0b3582ca9870346b989f6fac8814e80",
     "pointer_blob": "b368132e9d2772b51882fb164143d4c955506ded"},
    {"directory": "X2/Xp", "bytes": 6478, "md5": "1ef4ce1667283f97a566f505908b9a11",
     "pointer_blob": "461c49c12417693384a1e47610f9c50fb3d8a129"},
)))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def identity(subject):
    require(subject in SUBJECTS, "unselected_witness")
    item = WITNESSES[subject]
    key = f"MD5E-s{item['bytes']}--{item['md5']}.nii.gz"
    pointer = f"../../../.git/annex/objects/{item['directory']}/{key}/{key}".encode("ascii")
    require(len(pointer) == 137 and hashlib.sha1(b"blob 137\0" + pointer).hexdigest()
            == item["pointer_blob"], "fixed_pointer_identity_changed")
    return {**item, "participant": subject, "annex_key": key,
            "source_path": f"{subject}/ses-01/func/{subject}_ses-01_task-things_run-1_part-mag_space-T1w_desc-brain_mask.nii.gz",
            "url": f"{BASE}{item['directory']}/{key}/{key}"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Refusal("redirect_refused")


def verify_object(payload, expected):
    require(len(payload) == expected["bytes"] <= OBJECT_CAP, "witness_annex_length_mismatch")
    require(hashlib.md5(payload).hexdigest() == expected["md5"], "witness_annex_digest_mismatch")


def fetch_object(subject, report, deadline):
    expected = identity(subject)
    anatomy.check_deadline(deadline)
    require(report["network_requests"] < 4
            and report["transport_bytes_read"] + expected["bytes"] <= TOTAL_CAP,
            "witness_transport_bound")
    report["network_requests"] += 1
    access = {"participant": subject, "request_started_utc": utc_now(),
              "compressed_object_bytes_read": 0, "verified": False}
    report["accesses"].append(access)
    request = Request(expected["url"], headers={"Accept-Encoding": "identity",
                      "User-Agent": "EP17-fixed-anatomical-unit-witness"})
    opener = build_opener(NoRedirect())
    with opener.open(request, timeout=min(REQUEST_SECONDS, max(0.001, deadline-time.monotonic()))) as response:
        require(response.status == 200 and response.geturl() == expected["url"], "unexpected_witness_response")
        require(response.headers.get("Content-Encoding", "identity").lower() == "identity"
                and response.headers.get("Transfer-Encoding") is None, "encoded_witness_transport_refused")
        # Exact HTTP framing avoids reading an extra sentinel byte beyond the
        # adopted 28,069-byte total. A missing length remains unresolved.
        require(response.headers.get("Content-Length") == str(expected["bytes"]),
                "witness_content_length_mismatch")
        pieces, remaining = [], expected["bytes"]
        while remaining:
            anatomy.check_deadline(deadline)
            access.setdefault("compressed_bytes_first_read_attempted_utc", utc_now())
            block = response.read(remaining)
            require(len(block) <= remaining, "witness_transport_bound")
            if not block:
                break
            pieces.append(block)
            remaining -= len(block)
            access["compressed_object_bytes_read"] += len(block)
            report["transport_bytes_read"] += len(block)
        payload = b"".join(pieces)
    anatomy.check_deadline(deadline)
    verify_object(payload, expected)
    access["verified"] = True
    return payload, expected


def decode_header(payload):
    # max_length is an output limit: no voxel bytes are decoded, even if the
    # complete compressed mask was transported for annex authentication.
    decoder = zlib.decompressobj(wbits=31)
    block = decoder.decompress(payload, max_length=HEADER_BYTES)
    require(len(block) == HEADER_BYTES, "short_witness_header")
    header = nib.Nifti1Header(binaryblock=block[:348], check=False)
    require(int(header["sizeof_hdr"]) == 348 and bytes(header["magic"]) == b"n+1\0",
            "witness_not_single_file_nifti1")
    shape = header.get_data_shape()
    require(len(shape) == 3 and all(n > 0 for n in shape)
            and np.prod(shape) <= anatomy.MASK_VOXEL_CAP, "witness_shape_unresolved")
    require(np.isfinite(float(header["vox_offset"])) and float(header["vox_offset"]) >= HEADER_BYTES,
            "witness_payload_overlaps_header")
    require(header.get_xyzt_units()[0] == "mm", "witness_explicit_mm_required")
    return header


def recorded_union_header(saved):
    """Reconstruct only previously recorded geometry, never reopen the mask."""
    require(saved.get("voxel_values_read") is False, "unexpected_union_metadata_stage")
    header = nib.Nifti1Header()
    header.set_data_shape(saved["shape"])
    header["datatype"] = saved["datatype_code"]
    header["pixdim"] = saved["pixdim"]
    for name in ("qform_code", "sform_code"):
        header[name] = saved[name]
    for name, values in zip(("srow_x", "srow_y", "srow_z"), saved["srow_xyz"]):
        header[name] = values
    for name, value in zip(("quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z"),
                           saved["quaternion_and_offset"]):
        header[name] = value
    anatomy.verify_mask_header(header, saved)
    return header


def compare_geometry(header, saved):
    witness_affine, witness_geometry = anatomy.select_affine(header, anatomy.native_frame_authority(header))
    union = recorded_union_header(saved)
    require(tuple(union.get_data_shape()) == tuple(header.get_data_shape()), "witness_union_shape_mismatch")
    # Temporary mm labeling expresses the hypothesis to check against the
    # source's explicit mm; it does not modify or claim units in the source.
    union.set_xyzt_units("mm")
    union_affine, union_geometry = anatomy.select_affine(union, anatomy.native_frame_authority(union))
    union_geometry["reconstructed_units_for_comparison"] = union_geometry.pop("original_spatial_units")
    corners = np.array([(*point, 1) for point in itertools.product(
        *[(0, n-1) for n in header.get_data_shape()])], dtype=np.float64)
    difference = float(np.max(np.linalg.norm((corners @ (witness_affine-union_affine).T)[:, :3], axis=1)))
    require(difference <= anatomy.CORNER_TOLERANCE_MM, "witness_union_affine_mismatch")
    return {"witness_geometry": witness_geometry, "recorded_union_numeric_geometry": union_geometry,
            "union_numeric_geometry_unit_label_is_witness_derived": True,
            "maximum_corner_difference_mm": difference,
            "inherited_spatial_units": "mm", "union_header_rewritten": False}


def finish(report, deadline):
    require([p["participant"] for p in report["participants"]] == list(SUBJECTS)
            and report["network_requests"] == 4 and report["transport_bytes_read"] == TOTAL_CAP,
            "incomplete_witness_roster")
    report.update(status="completed", anatomical_spatial_units_witness_complete=True)
    anatomy.check_deadline(deadline)


def run_attempt(out):
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    report = {"analysis_id": "anatomical_axis", "mode": "unit_witness", "status": "running",
              "anatomical_spatial_units_witness_complete": False,
              "anatomical_axis_complete": False, "response_units_verified": False,
              "structural_voxel_arrays_decoded": False, "neural_values_read": False,
              "image_stimulus_values_read": False, "candidate_scoring_run": False,
              "audit_opened": False, "source_roles_complete": False,
              "compressed_structural_object_transport_selected": True,
              "decoded_bytes_per_object_maximum": HEADER_BYTES,
              "automatic_retries": 0, "redirects_followed": 0,
              "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 180,
              "fmriprep_commit": FMRIPREP_COMMIT, "dataset_id": DATASET_ID,
              "source_metadata": str(anatomy.METADATA), "source_masks_reopened": False,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
              "versions": {"numpy": np.__version__, "nibabel": nib.__version__},
              "network_requests": 0, "transport_bytes_read": 0,
              "accesses": [], "participants": [], "errors": []}
    old = {sig: signal.signal(sig, anatomy.deadline_alarm) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(INTERNAL_SECONDS)
    try:
        prerequisites = anatomy.load_metadata()
        for subject in SUBJECTS:
            payload, expected = fetch_object(subject, report, deadline)
            header = decode_header(payload)
            geometry = compare_geometry(header, prerequisites[subject]["mask_header"])
            report["participants"].append({**expected, **geometry})
            del payload, header
            anatomy.check_deadline(deadline)
        finish(report, deadline)
    except Refusal as error:
        report.update(status="stopped", anatomical_spatial_units_witness_complete=False)
        report["errors"].append(error.detail)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, MemoryError, HTTPError,
            URLError, zlib.error, nib.filebasedimages.ImageFileError) as error:
        report.update(status="stopped", anatomical_spatial_units_witness_complete=False)
        report["errors"].append({"code": "witness_processing_failure", "exception_type": type(error).__name__})
    finally:
        signal.alarm(0)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    report["elapsed_seconds"] = time.monotonic() - started
    if report["elapsed_seconds"] >= INTERNAL_SECONDS:
        report.update(status="stopped", anatomical_spatial_units_witness_complete=False)
        if not any(x["code"] == "attempt_deadline" for x in report["errors"]):
            report["errors"].append({"code": "attempt_deadline"})
    report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with (out / "unit_witness_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "anatomical_spatial_units_witness_complete", "network_requests", "transport_bytes_read")}))
    return 0 if report["status"] == "completed" and report["anatomical_spatial_units_witness_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(), "requires_coordinator_owned_compute")
    require(args.out.resolve() == OUTPUT.resolve() and not args.out.exists(), "fresh_selected_witness_attempt_required")
    args.out.mkdir(parents=True)
    return run_attempt(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
