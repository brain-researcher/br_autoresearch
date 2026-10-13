#!/usr/bin/env python3
"""EP15 all-participant anatomical intersection on a SUIT-derived3mm lattice.

Only forward SUIT XYZ fields, native cerebellar masks and GMc probability
images are opened. No beta/variance/functional values or participant-native
coordinates are exported. No role assignment or scientific margin is activated.
"""
import argparse
from datetime import datetime, timezone
import io
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import struct
import time
import zipfile

import numpy as np

from inspect_registration_headers import nifti_header


SUFFIXES = ("space-SUIT_xfm.nii", "desc-cereb_mask.nii", "label-GMc_probseg.nii")
SOURCE_RELATIVE = "data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0/source"
STRIDE = 3
GM_THRESHOLD = .5
PROBABILITY_ROUNDOFF = 1e-6
MAX_MEMBER_BYTES = 32 * 1024 * 1024
SCRATCH_BUDGET_BYTES = 2 * 1024 ** 3
GRID_KEYS = ("shape", "pixdim", "sform_code", "sform_rows", "qform_code",
             "quaternion_and_offsets")


def selected_sform(header):
    """Use the coded sform, as in the provider's NIfTI affine precedence."""
    if header["sform_code"] <= 0:
        raise ValueError("Expected the previously qualified coded sform; no fallback affine")
    if (header["xyzt_units"] & 7) != 2:
        raise ValueError("Anatomical spatial units are not explicitly millimetres; do not guess")
    affine = np.eye(4, dtype=np.float64)
    affine[:3] = header["sform_rows"]
    if not np.all(np.isfinite(affine)) or np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError("Nonfinite or singular anatomical sform")
    return affine


def read_image(archive, member, expected_header):
    """Read one exact NIfTI member, applying scaling only at sampled values."""
    info = archive.getinfo(member)
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError("Anatomical member exceeds the prior-header-derived32MiB bound")
    with archive.open(info) as stream:
        raw_header = stream.read(348)
        header = nifti_header(io.BytesIO(raw_header))
        for key in GRID_KEYS + ("datatype", "voxel_offset"):
            if header[key] != expected_header[key]:
                raise ValueError("Anatomical header differs from the completed all-participant header inspection")
        affine = selected_sform(header)
        endian = "<" if struct.unpack_from("<i", raw_header)[0] == 348 else ">"
        dtype_names = {2: "u1", 4: "i2", 16: "f4"}
        if header["datatype"] not in dtype_names:
            raise ValueError("Unexpected anatomical datatype")
        dtype = np.dtype(endian + dtype_names[header["datatype"]])
        bitpix = struct.unpack_from(endian + "h", raw_header, 72)[0]
        if bitpix != 8 * dtype.itemsize:
            raise ValueError("Anatomical datatype and bitpix disagree")
        slope, intercept = struct.unpack_from(endian + "2f", raw_header, 112)
        if slope == 0 or math.isnan(slope):
            slope, intercept = 1., 0.  # Defined NIfTI no-scaling convention.
        elif not math.isfinite(slope) or not math.isfinite(intercept):
            raise ValueError("Undefined anatomical intensity scaling")
        offset = header["voxel_offset"]
        if not math.isfinite(offset) or offset != int(offset) or not 352 <= offset <= 1024 * 1024:
            raise ValueError("Unsupported NIfTI payload offset")
        size = math.prod(header["shape"]) * dtype.itemsize
        if int(offset) + size > info.file_size:
            raise ValueError("Truncated anatomical image")
        if len(stream.read(int(offset) - 348)) != int(offset) - 348:
            raise ValueError("Truncated anatomical header extension")
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError("Truncated anatomical voxel payload")
    # NIfTI voxel i is the fastest-varying dimension: MATLAB/Fortran order.
    values = np.frombuffer(payload, dtype=dtype).reshape(header["shape"], order="F")
    return {"values": values, "header": header, "affine": affine,
            "slope": slope, "intercept": intercept, "payload_bytes": size}


def scale_values(image, values):
    return np.asarray(values, dtype=np.float64) * image["slope"] + image["intercept"]


def world_to_voxel(affine, world):
    inverse = np.linalg.inv(affine)
    return np.asarray(world, dtype=np.float64) @ inverse[:3, :3].T + inverse[:3, 3]


def strict_trilinear(image, voxel_coordinates, require_positive=False):
    """Require all positive-weight corners in bounds and finite; never renormalize.

    Zero-weight corners are neither read nor required, so an exact final-grid
    voxel can be valid. Invalid locations return NaN and false, never zero fill.
    require_positive implements the complete native cerebellar-mask stencil.
    """
    volume = image["values"]
    coordinates = np.asarray(voxel_coordinates, dtype=np.float64)
    if volume.ndim != 3 or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("A3-D image and N-by3 voxel coordinates are required")
    shape = np.asarray(volume.shape, dtype=np.int64)
    eligible = np.all(np.isfinite(coordinates), axis=1)
    eligible &= np.all((coordinates >= 0) & (coordinates <= shape - 1), axis=1)
    base = np.zeros(coordinates.shape, dtype=np.int64)
    fraction = np.zeros(coordinates.shape, dtype=np.float64)
    base[eligible] = np.floor(coordinates[eligible]).astype(np.int64)
    fraction[eligible] = coordinates[eligible] - base[eligible]
    valid = eligible.copy()
    result = np.zeros(len(coordinates), dtype=np.float64)
    for corner in itertools.product((0, 1), repeat=3):
        corner = np.asarray(corner, dtype=np.int64)
        weight = np.prod(np.where(corner, fraction, 1. - fraction), axis=1)
        active = eligible & (weight > 0)
        indices = base + corner
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        valid[active & ~inside] = False
        selected = np.flatnonzero(active & inside)
        if not len(selected):
            continue
        positions = indices[selected]
        values = scale_values(image, volume[positions[:, 0], positions[:, 1], positions[:, 2]])
        acceptable = np.isfinite(values)
        if require_positive:
            acceptable &= values > 0
        valid[selected[~acceptable]] = False
        result[selected] += weight[selected] * values
    result[~valid] = np.nan
    return result, valid


def lattice_from_header(header):
    if header["shape"] != [141, 95, 87, 1, 3]:
        raise ValueError("Unexpected common forward SUIT field shape")
    affine = selected_sform(header)
    if not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), 1., rtol=0, atol=1e-6):
        raise ValueError("Expected the qualified1mm forward-transform grid")
    axes = [np.arange(0, size, STRIDE, dtype=np.int16) for size in header["shape"][:3]]
    ijk = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    lattice_affine = affine.copy()
    lattice_affine[:3, :3] *= STRIDE
    return ijk, world, tuple(map(len, axes)), lattice_affine


def participant_support(images, lattice_indices):
    forward, mask, gray = (images[suffix] for suffix in SUFFIXES)
    ijk = lattice_indices
    pulled = scale_values(forward, forward["values"][ijk[:, 0], ijk[:, 1], ijk[:, 2], 0, :])
    finite_xyz = np.all(np.isfinite(pulled), axis=1)
    # GMc is a probability image, not an unscaled8-bit tissue label. A new
    # encoding discrepancy must fail, not silently produce another support.
    raw_gray = gray["values"]
    if raw_gray.ndim != 3 or mask["values"].ndim != 3:
        raise ValueError("Expected3-D cerebellar mask and GMc")
    finite_gray = raw_gray[np.isfinite(raw_gray)]
    if not len(finite_gray):
        raise ValueError("GMc has no finite values for probability-unit interpretation")
    raw_min, raw_max = np.min(finite_gray), np.max(finite_gray)
    del finite_gray
    endpoints = scale_values(gray, [raw_min, raw_max])
    if (not np.all(np.isfinite(endpoints)) or np.min(endpoints) < -PROBABILITY_ROUNDOFF
            or np.max(endpoints) > 1. + PROBABILITY_ROUNDOFF):
        raise ValueError("Scaled GMc does not have probability units; no implicit division/clipping")
    gm_values, gm_valid = strict_trilinear(gray, world_to_voxel(gray["affine"], pulled))
    unused_mask_values, mask_valid = strict_trilinear(mask, world_to_voxel(mask["affine"], pulled), require_positive=True)
    supported = finite_xyz & gm_valid & (gm_values >= GM_THRESHOLD) & mask_valid
    return supported, {
        "finite_forward_points": int(np.sum(finite_xyz)),
        "valid_GMc_sampling_points": int(np.sum(gm_valid)),
        "GMc_threshold_points": int(np.sum(gm_valid & (gm_values >= GM_THRESHOLD))),
        "valid_full_native_mask_stencils": int(np.sum(mask_valid)),
        "anatomically_supported_points": int(np.sum(supported)),
    }


def deadline(signum, frame):
    raise TimeoutError("Selected anatomical support deadline reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=540)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("The internal deadline must be between 1 and 540 seconds")
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use root's verified Slurm compute allocation")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Numeric thread counts must all be1")
    outputs = Path(__file__).resolve().parents[1]
    if outputs / "runs/source_qualification" not in args.output_dir.resolve().parents:
        raise ValueError("Output must be a new source_qualification attempt in this episode")
    prior = json.loads((outputs / "runs/source_qualification/attempt001/anatomical_headers.json").read_text())
    joined = json.loads((outputs / "runs/source_qualification/metadata_join001/source_metadata.json").read_text())
    people = sorted(prior["participants"], key=lambda row: row["participant_id"])
    ids = [row["participant_id"] for row in people]
    if (len(ids) != 24 or len(set(ids)) != 24 or not prior["forward_transform_headers_share_grid"]
            or not joined["metadata_join_complete"]
            or ids != sorted(row["participant_id"] for row in joined["participants"])):
        raise ValueError("Completed all24 source metadata and common forward grid are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    report = {
        "analysis_id": "source_qualification", "stage": "anatomical_common_support",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "specification": "outputs/anatomical_support_specification.md",
        "internal_deadline_seconds": args.max_seconds,
        "grid_name": "EP15_SUIT_derived_stride3_zero_based_not_canonical_SUIT3",
        "participant_ids": ids, "participants_completed": 0, "anatomical_members_read": 0,
        "anatomical_voxel_payload_bytes_read": 0, "beta_image_streams_opened": 0,
        "Task_B_values_read": False, "Task_A_reliability_computed": False,
        "participant_roles_assigned": False, "scientific_margins_activated": False,
        "individual_anatomy_or_native_coordinates_exported": False,
        "preparation_scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        "image_extraction_to_disk": False, "anatomical_intersection_complete": False,
        "functional_beta_support_qualified": False, "candidate_scoring_authorized": False,
    }
    counters, common = [], None
    reference_grid = None
    exit_code = 0
    try:
        for person in people:
            pid = person["participant_id"]
            source = Path(os.environ["OAK"]) / SOURCE_RELATIVE / f"MDTB_{pid}.zip"
            with zipfile.ZipFile(source) as archive:
                images = {}
                for suffix in SUFFIXES:
                    member = f"{pid}/anat/{pid}_{suffix}"
                    images[suffix] = read_image(archive, member, person["headers"][suffix])
                    report["anatomical_members_read"] += 1
                    report["anatomical_voxel_payload_bytes_read"] += images[suffix]["payload_bytes"]
            forward_header = images[SUFFIXES[0]]["header"]
            actual_grid = {key: forward_header[key] for key in GRID_KEYS}
            if common is None:
                reference_grid = actual_grid
                ijk, world, grid_shape, lattice_affine = lattice_from_header(forward_header)
                common = np.ones(len(ijk), dtype=bool)
                report["candidate_lattice_points"] = len(ijk)
                report["lattice_shape"] = grid_shape
                report["lattice_affine_mm"] = lattice_affine.tolist()
                report["scaling_header_values_seen"] = {suffix: [] for suffix in SUFFIXES}
            elif actual_grid != reference_grid:
                raise ValueError("Live forward-transform grids differ; do not resample a different grid")
            for suffix, image in images.items():
                pair = [image["slope"], image["intercept"]]
                if pair not in report["scaling_header_values_seen"][suffix]:
                    report["scaling_header_values_seen"][suffix].append(pair)
            support, counts = participant_support(images, ijk)
            common &= support
            counters.append(counts)
            del images, support
            report["participants_completed"] += 1
            if report["participants_completed"] == 1:
                report["first_participant_elapsed_seconds"] = time.monotonic() - started
                print(json.dumps({"participants_completed": 1, "elapsed_seconds": report["first_participant_elapsed_seconds"]}), flush=True)
        # Only an all24 intersection is exported; no partial/per-person masks.
        output = args.output_dir / "common_support.npz"
        np.savez_compressed(output, common_mask=common.reshape(grid_shape),
                            world_xyz_mm=world[common], source_transform_voxel_ijk=ijk[common],
                            lattice_affine_mm=lattice_affine)
        report["common_support_file"] = output.name
        report["common_support_points"] = int(np.sum(common))
        report["group_sampling_counts_summed_over_participants"] = {
            key: sum(row[key] for row in counters) for key in counters[0]}
        report["anatomical_intersection_complete"] = True
        report["nonempty_common_anatomical_support"] = bool(np.any(common))
        report["status"] = "complete_all24_anatomical_support" if np.any(common) else "complete_but_empty_anatomical_intersection"
        report["output_bytes"] = output.stat().st_size
        if not np.any(common):
            exit_code = 2
    except Exception as error:
        report.update(status="anatomical_support_execution_failure", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output_dir / "anatomical_support_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "participants_completed", "anatomical_members_read",
                                                 "elapsed_seconds", "max_rss_kib", "beta_image_streams_opened")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
