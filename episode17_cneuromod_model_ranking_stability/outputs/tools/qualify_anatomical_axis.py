#!/usr/bin/env python3
"""Map exactly eight adopted EP17 structural volumes; never open neural arrays."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import nibabel as nib
import numpy as np
import scipy
from scipy.ndimage import affine_transform


EPISODE = Path(__file__).resolve().parents[2]
SOURCE = EPISODE.parent.parent / "br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw"
METADATA = EPISODE / "outputs/runs/source_metadata/attempt003/metadata_report.json"
OUTPUT_PARENT = EPISODE / "outputs/runs/anatomical_axis"
UNIT_WITNESS = OUTPUT_PARENT / "units001/unit_witness_report.json"
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-06")
PRESENTATIONS = dict(zip(SUBJECTS, (12780, 12780, 12780, 11640)))
STAGES = {"B": "TYPEB_FITHRF.mat", "C": "TYPEC_FITHRF_GLMDENOISE.mat",
          "D": "TYPED_FITHRF_GLMDENOISE_RR.mat"}
ATLAS = dict(zip(SUBJECTS, (
    (1882667, "2ba3c3702f7b3239a5520995c4b5714a"),
    (1967036, "aff7cfe74e25391963fc46adf8059ecd"),
    (1935990, "f87c0b09c80c4c79ace45407167d8402"),
    (1756349, "6fe93a02e27c1fc6c590db53fd92c311"),
)))
UNIT_WITNESS_OBJECTS = dict(zip(SUBJECTS, (
    (7602, "aaf135ebac3bced281281b703b20fe9f"),
    (7127, "2510838dea8c0347033e769384730d38"),
    (6862, "f0b3582ca9870346b989f6fac8814e80"),
    (6478, "1ef4ce1667283f97a566f505908b9a11"),
)))
INTERNAL_SECONDS = 300
MASK_VOXEL_CAP = 4_000_000
ATLAS_VOXEL_CAP = 32_000_000
TABLE_BYTE_CAP = 256 * 1024 * 1024
METADATA_BYTE_CAP = 256 * 1024
CORNER_TOLERANCE_MM = 0.001
FIELDS = ("participant", "model_column_zero_based", "c_flat_index_zero_based",
          "i", "j", "k", "x_mm", "y_mm", "z_mm", "atlas_label", "nominated_hemisphere")
COMPLETION_FLAGS = ("structural_voxel_axis_complete", "anatomical_fusiform_mapping_complete")


class Refusal(Exception):
    def __init__(self, code, **details):
        super().__init__(code)
        self.detail = {"code": code, **details}


def require(condition, code, **details):
    if not condition:
        raise Refusal(code, **details)


def check_deadline(deadline):
    require(time.monotonic() < deadline, "attempt_deadline")


def deadline_alarm(signum, frame):
    raise Refusal("attempt_deadline" if signum == signal.SIGALRM else "attempt_terminated")


def load_metadata():
    require(METADATA.stat().st_size <= METADATA_BYTE_CAP, "metadata_byte_cap")
    with METADATA.open("rb") as stream:
        raw = stream.read(METADATA_BYTE_CAP + 1)
    require(len(raw) <= METADATA_BYTE_CAP, "metadata_byte_cap")
    report = json.loads(raw)
    require(report.get("analysis_id") == "source_metadata" and not report.get("structural_errors"),
            "metadata_prerequisite_incomplete")
    require(report.get("source") == str(SOURCE), "metadata_source_changed")
    participants = report.get("participants", [])
    require([p.get("participant") for p in participants] == list(SUBJECTS), "participant_roster_changed")
    result = {}
    for participant in participants:
        subject = participant["participant"]
        require(participant.get("presentations") == PRESENTATIONS[subject]
                and participant.get("B_C_D_stored_shapes_equal") is True
                and not participant.get("structural_issues"), "BCD_header_prerequisite_incomplete")
        axes = participant["stage_axes"]
        count = None
        for stage, filename in STAGES.items():
            axis = axes[stage]
            shape = axis.get("stored_shape", [])
            require(len(shape) == 4 and all(type(n) is int and n > 0 for n in shape)
                    and shape[:3] == [PRESENTATIONS[subject], 1, 1]
                    and shape[3] <= MASK_VOXEL_CAP, "BCD_axis_changed")
            require(axis.get("file") == f"THINGS/glmsingle/{subject}/glmsingle/output/T1w/{filename}"
                    and axis.get("dataset") == "modelmd" and axis.get("dtype") == "float32"
                    and axis.get("trial_axis_matches_design_count") is True, "BCD_identity_changed")
            require(count is None or count == shape[3], "BCD_voxel_counts_differ")
            count = shape[3]
        result[subject] = {"voxel_count": count, "mask_header": participant["mask_headers"]["union"]}
    return result


def source_path(subject, kind):
    require(subject in SUBJECTS and kind in ("mask", "atlas"), "unselected_source_object")
    if kind == "mask":
        return SOURCE / f"THINGS/glmsingle/{subject}/glmsingle/input/{subject}_task-things_space-T1w_label-brain_desc-union_mask.nii"
    return SOURCE / f"anatomical/smriprep/{subject}/anat/{subject}_desc-aparcaseg_dseg.nii.gz"


def original_file(path):
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(SOURCE.resolve()) and resolved.is_file(), "source_object_leaves_release")
    return resolved


def verify_atlas_object(path, expected_size, expected_md5, deadline):
    require(path.stat().st_size == expected_size, "atlas_annex_length_mismatch")
    digest = hashlib.md5()
    count = 0
    with path.open("rb") as stream:
        while True:
            check_deadline(deadline)
            chunk = stream.read(min(65536, expected_size + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            require(count <= expected_size, "atlas_annex_length_mismatch")
            digest.update(chunk)
    require(count == expected_size and digest.hexdigest() == expected_md5, "atlas_annex_identity_mismatch")


def read_volume(path, compressed, voxel_cap, deadline):
    """Read NIfTI-1 header before its bounded payload; honor on-disk F ordering.

    NiBabel owns dtype, scaling and qform semantics. Its header constructor
    reads no extensions or adjacent files; this selected stage uses a single
    explicitly opened .nii/.nii.gz stream.
    """
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        raw_header = stream.read(348)
        require(len(raw_header) == 348, "short_nifti_header")
        header = nib.Nifti1Header(binaryblock=raw_header, check=False)
        require(int(header["sizeof_hdr"]) == 348 and bytes(header["magic"]) == b"n+1\x00", "not_single_file_nifti1")
        shape = tuple(int(n) for n in header.get_data_shape())
        require(len(shape) == 3 and all(n > 0 for n in shape), "volume_must_be_3d")
        count = math.prod(shape)
        dtype = header.get_data_dtype()
        require(count <= voxel_cap and dtype.kind in "iuf" and dtype.itemsize <= 8,
                "structural_volume_decode_bound")
        require(int(header["bitpix"]) == dtype.itemsize * 8, "nifti_bitpix_mismatch")
        offset = float(header["vox_offset"])
        require(math.isfinite(offset) and offset == int(offset) and 352 <= offset <= 1024 * 1024,
                "nifti_data_offset_unresolved")
        # Bound extension/padding consumption; no extension contents are exported.
        skip = int(offset) - 348
        require(len(stream.read(skip)) == skip, "short_nifti_padding")
        check_deadline(deadline)
        size = count * dtype.itemsize
        payload = stream.read(size)
        require(len(payload) == size, "short_nifti_payload")
    values = np.frombuffer(payload, dtype=dtype).reshape(shape, order="F").astype(np.float64)
    slope, intercept = header.get_slope_inter()
    if slope is not None:
        values *= slope
        values += intercept
    check_deadline(deadline)
    return header, values


def native_frame_authority(header):
    """Use the adopted exact-file native-T1w provenance, never image overlap.

    Codes 1/2 describe anatomical coordinates, while 3/4/5 denote standard
    spaces. Different active codes need additional per-form source evidence;
    the eight-file selection does not invent that evidence or expose a CLI
    override. No coded form is an unresolved source, not a guessed fallback.
    """
    codes = {name: int(header[f"{name}_code"]) for name in ("qform", "sform")}
    active = {name: code for name, code in codes.items() if code != 0}
    require(active and set(active.values()).issubset({1, 2}), "native_T1w_form_unresolved")
    require(len(set(active.values())) == 1, "different_form_frames_need_source_evidence")
    return {name: "T1w" for name in active}


def select_affine(header, source_frames):
    """Only source-authenticated T1w forms qualify; no automatic best-affine."""
    units = header.get_xyzt_units()[0]
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    require(units in factors, "spatial_units_unresolved")
    forms, codes = {}, {}
    for name in ("qform", "sform"):
        code = int(header[f"{name}_code"])
        codes[name] = code
        if code:
            matrix = np.asarray(getattr(header, f"get_{name}")(), dtype=np.float64)
            require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
                    and np.array_equal(matrix[3], [0, 0, 0, 1])
                    and np.linalg.det(matrix[:3, :3]) != 0, "invalid_coded_affine", form=name)
            matrix = matrix.copy()
            matrix[:3, :] *= factors[units]
            forms[name] = matrix
    eligible = [name for name in ("sform", "qform")
                if name in forms and source_frames.get(name) == "T1w"]
    require(eligible, "no_authenticated_T1w_affine")
    disagreement = None
    if len(eligible) == 2:
        corners = np.array([(*point, 1) for point in itertools.product(
            *[(0, n - 1) for n in header.get_data_shape()])], dtype=np.float64)
        differences = corners @ (forms["sform"] - forms["qform"]).T
        disagreement = float(np.max(np.linalg.norm(differences[:, :3], axis=1)))
        require(disagreement <= CORNER_TOLERANCE_MM, "authenticated_affines_disagree")
    chosen = eligible[0]
    matrix = forms[chosen]
    geometry = {"shape": list(header.get_data_shape()), "original_spatial_units": units,
                "unit_to_mm": factors[units], "qform_code": codes["qform"], "sform_code": codes["sform"],
                "coded_affines_mm": {name: affine.tolist() for name, affine in forms.items()},
                "source_authenticated_frames": source_frames, "selected_form": chosen,
                "selected_affine_mm": matrix.tolist(), "orientation": list(nib.aff2axcodes(matrix)),
                "eligible_form_corner_difference_mm": disagreement}
    return matrix, geometry


def verify_mask_header(header, expected):
    require(list(header.get_data_shape()) == expected.get("shape")
            and int(header["datatype"]) == expected.get("datatype_code")
            and int(header["qform_code"]) == expected.get("qform_code")
            and int(header["sform_code"]) == expected.get("sform_code"), "original_union_header_changed")
    require(np.array_equal(np.asarray(header["pixdim"]), expected.get("pixdim"))
            and np.array_equal(np.array([header[name] for name in ("srow_x", "srow_y", "srow_z")]),
                               expected.get("srow_xyz")), "original_union_geometry_changed")
    quaternion = [float(header[name]) for name in ("quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z")]
    require(quaternion == expected.get("quaternion_and_offset"), "original_union_qform_changed")


def load_unit_witnesses():
    """Consume the one completed metadata witness; perform no network/source I/O."""
    require(UNIT_WITNESS.stat().st_size <= METADATA_BYTE_CAP, "unit_witness_byte_cap")
    with UNIT_WITNESS.open("rb") as stream:
        payload = stream.read(METADATA_BYTE_CAP + 1)
    require(len(payload) <= METADATA_BYTE_CAP, "unit_witness_byte_cap")
    report = json.loads(payload)
    require(report.get("analysis_id") == "anatomical_axis" and report.get("mode") == "unit_witness"
            and report.get("status") == "completed"
            and report.get("anatomical_spatial_units_witness_complete") is True
            and report.get("errors") == [], "unit_witness_incomplete")
    require(report.get("source_metadata") == str(METADATA)
            and report.get("fmriprep_commit") == "5155ba0d43836a4451141b90b5a70507f41205cf"
            and report.get("dataset_id") == "81d8c126-b10f-4e8b-9eab-6825964c8552",
            "unit_witness_source_binding_changed")
    require(report.get("network_requests") == 4 and report.get("transport_bytes_read") == 28069
            and report.get("automatic_retries") == 0 and report.get("redirects_followed") == 0
            and report.get("decoded_bytes_per_object_maximum") == 352
            and report.get("structural_voxel_arrays_decoded") is False,
            "unit_witness_access_scope_changed")
    participants, accesses = report.get("participants", []), report.get("accesses", [])
    require([p.get("participant") for p in participants] == list(SUBJECTS)
            and [a.get("participant") for a in accesses] == list(SUBJECTS), "unit_witness_roster_changed")
    for item, access in zip(participants, accesses):
        subject = item["participant"]
        size, digest = UNIT_WITNESS_OBJECTS[subject]
        require(item.get("bytes") == size and item.get("md5") == digest
                and item.get("annex_key") == f"MD5E-s{size}--{digest}.nii.gz"
                and item.get("source_path") == f"{subject}/ses-01/func/{subject}_ses-01_task-things_run-1_part-mag_space-T1w_desc-brain_mask.nii.gz"
                and access.get("verified") is True and access.get("compressed_object_bytes_read") == size,
                "unit_witness_object_changed")
        geometry = item.get("witness_geometry", {})
        difference = item.get("maximum_corner_difference_mm")
        require(geometry.get("original_spatial_units") == "mm" and geometry.get("unit_to_mm") == 1
                and item.get("inherited_spatial_units") == "mm" and item.get("union_header_rewritten") is False
                and item.get("union_numeric_geometry_unit_label_is_witness_derived") is True
                and isinstance(difference, (float, int)) and math.isfinite(difference)
                and 0 <= difference <= CORNER_TOLERANCE_MM, "unit_witness_geometry_unqualified")
    return {item["participant"]: item for item in participants}


def select_union_affine(header, subject, saved_header, witness_cache):
    """Repair only unknown units in memory, retaining the observed source units."""
    verify_mask_header(header, saved_header)
    observed = header.get_xyzt_units()[0]
    frames = native_frame_authority(header)
    if observed != "unknown":
        affine, geometry = select_affine(header, frames)
        geometry.update(spatial_unit_authority="explicit_source_header", unit_witness_used=False)
        return affine, geometry
    if not witness_cache:
        witness_cache.update(load_unit_witnesses())
    require(subject in witness_cache, "unit_witness_participant_missing")
    evidence = witness_cache[subject]
    repaired = header.copy()
    repaired.set_xyzt_units("mm", header.get_xyzt_units()[1])
    affine, geometry = select_affine(repaired, frames)
    reconstructed = dict(geometry)
    reconstructed["reconstructed_units_for_comparison"] = reconstructed.pop("original_spatial_units")
    require(reconstructed == evidence.get("recorded_union_numeric_geometry"), "unit_witness_current_union_geometry_changed")
    witness = evidence["witness_geometry"]
    require(witness.get("shape") == list(header.get_data_shape()), "unit_witness_current_shape_changed")
    matrix = np.asarray(witness.get("selected_affine_mm"), dtype=np.float64)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), "unit_witness_invalid_affine")
    corners = np.array([(*point, 1) for point in itertools.product(
        *[(0, n - 1) for n in header.get_data_shape()])], dtype=np.float64)
    difference = float(np.max(np.linalg.norm((corners @ (affine - matrix).T)[:, :3], axis=1)))
    require(difference <= CORNER_TOLERANCE_MM, "unit_witness_current_affine_mismatch")
    geometry.update(original_spatial_units=observed, interpreted_spatial_units="mm",
                    spatial_unit_authority="pinned_first_source_mask_header_and_preserved_union_affine",
                    unit_witness_used=True, unit_witness_report=str(UNIT_WITNESS),
                    unit_witness_corner_difference_mm=difference, source_header_rewritten=False)
    return affine, geometry


def validate_values(mask, atlas, expected_columns):
    require(np.isfinite(mask).all() and np.logical_or(mask == 0, mask == 1).all(), "mask_not_binary_finite")
    require(int(np.count_nonzero(mask)) == expected_columns, "mask_BCD_column_count_mismatch")
    require(np.isfinite(atlas).all() and np.equal(atlas, np.floor(atlas)).all(), "atlas_not_finite_integer_labels")
    require(atlas.min() >= np.iinfo(np.int32).min and atlas.max() <= np.iinfo(np.int32).max,
            "atlas_label_outside_exact_int32")
    return mask.astype(bool), atlas.astype(np.int32)


def map_columns(mask, atlas, mask_affine, atlas_affine):
    transform = np.linalg.inv(atlas_affine) @ mask_affine
    mapped = affine_transform(atlas, transform[:3, :3], offset=transform[:3, 3],
                              output_shape=mask.shape, output=np.int32,
                              order=0, mode="constant", cval=0, prefilter=False)
    flat = np.flatnonzero(mask.ravel(order="C"))
    coordinates = np.column_stack(np.unravel_index(flat, mask.shape, order="C"))
    labels = mapped.ravel(order="C")[flat]
    world = coordinates @ mask_affine[:3, :3].T + mask_affine[:3, 3]
    atlas_coordinates = coordinates @ transform[:3, :3].T + transform[:3, 3]
    outside = np.any((atlas_coordinates < 0) | (atlas_coordinates > np.array(atlas.shape) - 1), axis=1)
    return flat, coordinates, world, labels, outside


def roi_summary(atlas, labels, world):
    result = {}
    for name, number in (("left", 1007), ("right", 2007)):
        selected = labels == number
        count = int(np.count_nonzero(selected))
        result[name] = {"label": number, "source_label_voxels": int(np.count_nonzero(atlas == number)),
                        "original_axis_voxels": count,
                        "world_min_mm": world[selected].min(axis=0).tolist() if count else None,
                        "world_max_mm": world[selected].max(axis=0).tolist() if count else None,
                        "refusal": "empty_nominated_ROI" if count == 0 else
                                   "single_voxel_cannot_change_support" if count == 1 else None,
                        "scientific_support_adequacy_verified": False}
    return result


class BoundedTable:
    def __init__(self, stream, cap=TABLE_BYTE_CAP):
        self.stream, self.cap = stream, cap
        self.bytes, self.rows = 0, 0

    def write(self, rows, header=False):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        if header:
            writer.writerow(FIELDS)
        writer.writerows(rows)
        data = buffer.getvalue().encode("utf-8")
        require(self.bytes + len(data) <= self.cap, "uncompressed_table_byte_cap")
        self.stream.write(data)
        self.bytes += len(data)
        self.rows += len(rows)


def write_columns(table, subject, mapped, deadline):
    flat, coordinates, world, labels, _ = mapped
    for start in range(0, len(flat), 4096):
        check_deadline(deadline)
        rows = []
        for column in range(start, min(start + 4096, len(flat))):
            number = int(labels[column])
            rows.append((subject, column, int(flat[column]), *map(int, coordinates[column]),
                         *[format(float(value), ".17g") for value in world[column]], number,
                         "left" if number == 1007 else "right" if number == 2007 else ""))
        table.write(rows)


def process_participant(subject, prerequisite, table, report, deadline, witness_cache=None):
    mask_path, atlas_path = source_path(subject, "mask"), source_path(subject, "atlas")
    mask_file, atlas_file = original_file(mask_path), original_file(atlas_path)
    size, digest = ATLAS[subject]
    verify_atlas_object(atlas_file, size, digest, deadline)
    report["verified_atlas_objects"].append(subject)
    mask_header, mask_values = read_volume(mask_file, False, MASK_VOXEL_CAP, deadline)
    report["structural_payloads_read"].append(str(mask_path.relative_to(SOURCE)))
    mask_affine, mask_geometry = select_union_affine(mask_header, subject, prerequisite["mask_header"],
                                                    {} if witness_cache is None else witness_cache)
    atlas_header, atlas_values = read_volume(atlas_file, True, ATLAS_VOXEL_CAP, deadline)
    report["structural_payloads_read"].append(str(atlas_path.relative_to(SOURCE)))
    atlas_affine, atlas_geometry = select_affine(atlas_header, native_frame_authority(atlas_header))
    mask, atlas = validate_values(mask_values, atlas_values, prerequisite["voxel_count"])
    del mask_values, atlas_values
    check_deadline(deadline)
    mapped = map_columns(mask, atlas, mask_affine, atlas_affine)
    summary = {"participant": subject, "original_model_columns": len(mapped[0]),
               "mask": str(mask_path.relative_to(SOURCE)), "atlas": str(atlas_path.relative_to(SOURCE)),
               "atlas_annex_bytes": size, "atlas_annex_md5": digest,
               "mask_geometry": mask_geometry, "atlas_geometry": atlas_geometry,
               "BCD_header_column_counts_agree": True,
               "out_of_FOV_mask_columns": int(np.count_nonzero(mapped[4])),
               "ROIs": roi_summary(atlas, mapped[3], mapped[2])}
    report["participants"].append(summary)
    write_columns(table, subject, mapped, deadline)
    summary["axis_rows_written"] = len(mapped[0])
    for hemisphere, roi in summary["ROIs"].items():
        if roi["refusal"]:
            report["refusals"].append({"participant": subject, "hemisphere": hemisphere, "code": roi["refusal"]})


def clear_completion(report):
    for key in COMPLETION_FLAGS:
        report[key] = False


def complete(report, deadline):
    report.update({"status": "completed", **{key: True for key in COMPLETION_FLAGS}})
    check_deadline(deadline)


def run_attempt(out):
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    report = {"analysis_id": "anatomical_axis", "status": "running",
              **{key: False for key in COMPLETION_FLAGS},
              "finite_response_support_complete": False, "response_units_verified": False,
              "reliability_support_complete": False, "source_roles_complete": False,
              "neural_values_read": False, "image_stimulus_values_read": False,
              "provider_response_masks_read": False, "provider_noise_ceilings_read": False,
              "DESIGNINFO_read": False, "candidate_scoring_run": False, "audit_opened": False,
              "support_N_selected": False, "network_requests": 0, "automatic_retries": 0,
              "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 360,
              "source_metadata": str(METADATA), "source_root": str(SOURCE),
              "mask_column_rule": "original_union_boolean_C_order",
              "modelmd_coordinate_rule": "modelmd[p,0,0,j]",
              "sampling": {"implementation": "scipy.ndimage.affine_transform", "order": 0,
                           "mode": "constant", "cval": 0, "prefilter": False,
                           "boundary": "outside continuous [0, shape-1] becomes zero; nearest tie follows recorded SciPy"},
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "nibabel": nib.__version__},
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
              "participants": [], "verified_atlas_objects": [], "structural_payloads_read": [],
              "refusals": [], "errors": []}
    previous = {sig: signal.signal(sig, deadline_alarm) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(INTERNAL_SECONDS)
    table = None
    try:
        prerequisites = load_metadata()
        witness_cache = {}
        with (out / "voxel_axis.csv.gz").open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
                table = BoundedTable(compressed)
                table.write([], header=True)
                for subject in SUBJECTS:
                    process_participant(subject, prerequisites[subject], table, report, deadline, witness_cache)
        check_deadline(deadline)
        require([x["participant"] for x in report["participants"]] == list(SUBJECTS), "incomplete_structural_roster")
        require(table.rows == sum(p["voxel_count"] for p in prerequisites.values()), "incomplete_axis_table")
        if report["refusals"]:
            report["status"] = "refused"
        else:
            complete(report, deadline)
    except Refusal as error:
        clear_completion(report)
        report["status"] = "stopped"
        report["errors"].append(error.detail)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, MemoryError, nib.filebasedimages.ImageFileError) as error:
        clear_completion(report)
        report["status"] = "stopped"
        report["errors"].append({"code": "structural_processing_failure", "exception_type": type(error).__name__})
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    report["elapsed_seconds"] = time.monotonic() - started
    if report["elapsed_seconds"] >= INTERNAL_SECONDS:
        clear_completion(report)
        report["status"] = "stopped"
        if not any(x["code"] == "attempt_deadline" for x in report["errors"]):
            report["errors"].append({"code": "attempt_deadline"})
    report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["table_rows_written"] = 0 if table is None else table.rows
    report["table_uncompressed_bytes_written"] = 0 if table is None else table.bytes
    report["partial_table_requires_complete_terminal_report"] = True
    # All source work and bounded CSV publication are complete before disabling
    # the alarm. The small terminal JSON contains no source arrays.
    with (out / "anatomical_axis_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", *COMPLETION_FLAGS, "table_rows_written")}))
    return 0 if report["status"] == "completed" and all(report[key] for key in COMPLETION_FLAGS) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(), "requires_coordinator_owned_compute")
    out = args.out.resolve()
    require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_episode_attempt_required")
    out.mkdir(parents=True)
    return run_attempt(out)


if __name__ == "__main__":
    raise SystemExit(main())
