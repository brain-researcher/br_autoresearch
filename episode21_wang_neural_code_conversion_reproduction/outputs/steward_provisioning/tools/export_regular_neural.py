#!/usr/bin/env python3
"""Bounded, sealed regular-response provisioning; requires separate approval."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np
from native_grid_binding import compare_native_grids

BASE = Path(__file__).resolve().parents[3]
PROV = BASE / "outputs/steward_provisioning"
ROLES = ("subject_unique", "tau_train", "tau_test")
FIELDS = ("participant", "session", "run", "beta_index", "beta_volume_index", "image_id", "role")
CAP = 10_000_000_000
CHUNK = 1 << 20
RESOURCES = {"jobs": 1, "cpus": 1, "memory_gib": 4, "minutes": 720,
             "gpus": 0, "durable_bytes": CAP, "scratch_bytes": CAP}


def require(condition, code):
    if not condition:
        raise RuntimeError(code)  # Codes only; no response values in exceptions.


def load_json(path):
    return json.loads(path.read_text())


class Guard:
    def __init__(self):
        self.job = os.environ.get("SLURM_JOB_ID")
        require(self.job is not None, "slurm_allocation_required")
        auth = load_json(BASE / "outputs/regular_neural_export_authorization_2026-09-30.json")
        require(auth.get("approved") is True and auth.get("purpose") == "regular_neural_steward_export", "explicit_export_approval_required")
        require(auth.get("resources") == RESOURCES, "allocation_mismatch")
        require(auth.get("opaque_mixed_stream_traversal_permitted") is True, "stream_traversal_approval_required")
        require(all(auth.get(k) is False for k in ("allow_ood_arrays", "allow_fitting", "allow_scoring")), "boundary_mismatch")
        amendment = load_json(BASE / "outputs/protocol_provider_beta_amendment_2026-09-30.yaml")
        require(amendment["status"] == "scientist_approved_adopted", "measurement_amendment_required")
        require(int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) == 1, "CPU_allocation_mismatch")
        require(int(os.environ.get("SLURM_MEM_PER_NODE", "0")) == 4096, "memory_allocation_mismatch")
        claim = PROV / "regular_neural_export_job_claim.json"
        with claim.open("x") as f:
            json.dump({"job_id": self.job, "purpose": auth["purpose"]}, f)
        self.deadline = time.monotonic() + 720 * 60 - 120
        self.source_root = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922/dataset"
        inventory = load_json(self.source_root.parent / "provider_manifests/glmsingle.json")
        self.object_sizes = {r["Key"]: r["Size"] for r in inventory["Contents"]}
        self.durable_bytes = self.scratch_bytes = 0
        self.bindings = []

    def check(self):
        require(time.monotonic() < self.deadline, "export_deadline_reached")

    def reserve(self, count, durable=False):
        self.check()
        attr = "durable_bytes" if durable else "scratch_bytes"
        new = getattr(self, attr) + count
        require(new <= CAP, "durable_cap_exceeded" if durable else "scratch_cap_exceeded")
        setattr(self, attr, new)

    def source(self, path):
        path = Path(path).resolve()
        key = path.relative_to(self.source_root.resolve()).as_posix()
        require(path.stat().st_size == self.object_sizes[key], "source_object_size_mismatch")
        return path


def read_exact(stream, count, guard):
    parts = []
    while count:
        guard.check()
        part = stream.read(min(count, CHUNK))
        require(bool(part), "truncated_source")
        parts.append(part)
        count -= len(part)
    return b"".join(parts)


def discard(stream, count, guard):
    while count:
        guard.check()
        part = stream.read(min(count, CHUNK))
        require(bool(part), "truncated_source")
        count -= len(part)  # Opaque bytes; never decode excluded samples.


class Writer:
    def __init__(self, directory, role, count, voxels, guard):
        self.guard, self.count, self.rows, self.voxels = guard, count, 0, voxels
        self.files = []
        self.expected_lengths = {}
        self.response = self.array(directory / f"{role}_responses.npy", count, voxels, "<f4")
        self.finite = self.array(directory / f"{role}_finite_packed.npy", count, (voxels + 7) // 8, "|u1")
        self.tsv = (directory / f"{role}_trials.tsv").open("wb")
        self.files.append(self.tsv)
        self.write(self.tsv, ("output_row_index\t" + "\t".join(FIELDS) + "\n").encode())
        self.nonfinite = 0

    def array(self, path, rows, cols, dtype):
        handle = path.open("wb")
        buf = io.BytesIO()
        np.lib.format.write_array_header_2_0(buf, {"descr": dtype, "fortran_order": False, "shape": (rows, cols)})
        self.write(handle, buf.getvalue())
        self.expected_lengths[handle] = len(buf.getvalue()) + rows * cols * np.dtype(dtype).itemsize
        self.files.append(handle)
        return handle

    def write(self, handle, data):
        self.guard.reserve(len(data))
        handle.write(data)

    def append(self, values, row):
        require(self.rows < self.count and values.shape == (self.voxels,), "output_row_shape_mismatch")
        finite = np.isfinite(values)
        self.write(self.response, values.tobytes(order="C"))
        self.write(self.finite, np.packbits(finite, bitorder="little").tobytes())
        require(all("\t" not in row[k] and "\n" not in row[k] and "\r" not in row[k] for k in FIELDS), "invalid_trial_identifier")
        self.write(self.tsv, (str(self.rows) + "\t" + "\t".join(row[k] for k in FIELDS) + "\n").encode())
        self.nonfinite += int(self.voxels - np.count_nonzero(finite))
        self.rows += 1

    def close(self):
        require(self.rows == self.count, "incomplete_role_rows")
        for handle in self.files:
            if handle in self.expected_lengths:
                require(handle.tell() == self.expected_lengths[handle], "output_array_length_mismatch")
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        return {"rows": self.rows, "voxels": self.voxels, "nonfinite_elements": self.nonfinite,
                "finite_mask_bitorder": "little", "dtype": "float32", "repetitions_preserved": True}


def session_export(session, rows, subject, storage, writers, guard):
    shape = tuple(session["header"]["shape"])
    require(len(shape) == 4 and shape[:3] == tuple(subject["grid"]["shape"]), "expected_grid_mismatch")
    selected = {}
    for row in rows:
        ordinal = int(row["beta_volume_index"])
        require(row["role"] in ROLES and ordinal not in selected and 0 <= ordinal < shape[3], "invalid_trial_binding")
        require(row["participant"] == session["participant"] and row["session"] == session["session"], "trial_session_mismatch")
        selected[ordinal] = row
    require(len(rows) == session["regular_rows"], "session_regular_count_mismatch")
    with gzip.open(guard.source(session["beta_path"]), "rb") as stream:
        h = nib.Nifti1Header.from_fileobj(io.BytesIO(read_exact(stream, 348, guard)), check=False)
        require(h["magic"].tobytes() == b"n+1\x00", "not_single_file_NIfTI1")
        require(int(h["dim"][0]) == 4 and tuple(h.get_data_shape()) == shape, "source_shape_mismatch")
        require(int(h["datatype"]) == 16 and int(h["bitpix"]) == 32, "source_not_float32")
        dtype = np.dtype(h.get_data_dtype())
        binding = compare_native_grids({"shape": list(shape[:3]), "affine": h.get_best_affine().tolist(),
                                        "spatial_units": h.get_xyzt_units()[0]}, subject["grid"])
        require(binding["consistent"], "native_grid_mismatch")
        offset = float(h["vox_offset"])
        require(math.isfinite(offset) and offset.is_integer() and 352 <= offset <= CAP, "invalid_data_offset")
        discard(stream, int(offset) - 348, guard)
        slope, intercept = h.get_slope_inter()
        if slope is None:
            slope, intercept = 1.0, 0.0
        require(math.isfinite(slope) and math.isfinite(intercept), "invalid_header_scaling")
        guard.bindings.append({"participant": session["participant"], "session": session["session"],
                               "shape": list(shape), "endian": h.endianness, "offset": int(offset),
                               "header_scaling": [float(slope), float(intercept)], "native_grid": binding})
        nbytes = math.prod(shape[:3]) * dtype.itemsize
        for ordinal in range(shape[3]):
            row = selected.get(ordinal)
            if row is None:
                discard(stream, nbytes, guard)
                continue
            raw = read_exact(stream, nbytes, guard)
            values = np.frombuffer(raw, dtype=dtype)[storage]
            with np.errstate(over="ignore", invalid="ignore"):
                if slope != 1.0 or intercept != 0.0:
                    values = (values.astype(np.float64) * slope + intercept).astype("<f4")
                else:
                    values = values.astype("<f4", copy=False)
            writers[row["role"]].append(values, row)
        # Consume remaining opaque trailer bytes and force gzip CRC validation.
        while stream.read(CHUNK):
            guard.check()


def publish_subject(stage, building, guard):
    target = building / stage.name
    target.mkdir(mode=0o700)
    for source in sorted(stage.iterdir()):
        size = source.stat().st_size
        guard.reserve(size, durable=True)
        with source.open("rb") as src, (target / source.name).open("xb") as dst:
            copied = 0
            while part := src.read(CHUNK):
                guard.check()
                dst.write(part)
                copied += len(part)
            require(copied == size, "publication_size_mismatch")
            dst.flush()
            os.fsync(dst.fileno())
        (target / source.name).chmod(0o400)
    target.chmod(0o500)
    removed = sum(p.stat().st_size for p in stage.iterdir())
    shutil.rmtree(stage)
    guard.scratch_bytes -= removed


def main():
    os.umask(0o077)
    guard = Guard()  # All authorization checks precede any beta payload access.
    destination = PROV / "regular_neural_sealed"
    report_path = PROV / "regular_neural_structural_report.json"
    require(not destination.exists() and not report_path.exists(), "existing_export_do_not_overwrite")
    building = PROV / f"regular_neural_sealed.building.{guard.job}"
    building.mkdir(mode=0o700)
    scratch = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / f"neural-export-{guard.job}"
    scratch.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifest = load_json(PROV / "source_structural_manifest.json")
    masks = load_json(PROV / "independent_visual_masks/mask_report.json")
    grouped = {}
    with (PROV / "regular_trial_index.tsv").open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        require(tuple(reader.fieldnames) == FIELDS, "trial_columns_mismatch")
        for row in reader:
            require(row["role"] in ROLES, "forbidden_trial_role")
            grouped.setdefault((row["participant"], row["session"]), []).append(row)
    summaries = []
    for subject in sorted(masks["subjects"], key=lambda s: s["participant"]):
        participant = subject["participant"]
        mask_file = PROV / "independent_visual_masks" / subject["voxel_indices_file"]
        require(hashlib.sha256(mask_file.read_bytes()).hexdigest() == subject["voxel_indices_sha256"], "mask_identity_mismatch")
        indices = np.load(mask_file, allow_pickle=False)
        require(indices.ndim == 1 and indices.dtype.kind in "iu" and len(indices) == subject["union_voxel_count"], "mask_index_shape_mismatch")
        require(np.all(indices[1:] > indices[:-1]) and int(indices[0]) >= 0 and int(indices[-1]) < math.prod(subject["grid"]["shape"]), "invalid_mask_indices")
        i, j, k = np.unravel_index(indices, subject["grid"]["shape"], order="C")
        nx, ny, _ = subject["grid"]["shape"]
        storage = i + nx * j + nx * ny * k
        sessions = sorted((s for s in manifest["observed_sessions"] if s["participant"] == participant), key=lambda s: s["session"])
        require(len(sessions) == 30, "required_subject_sessions_missing")
        counts = {role: sum(row["role"] == role for s in sessions for row in grouped[(participant, s["session"])]) for role in ROLES}
        require(all(counts.values()), "empty_required_role")
        stage = scratch / participant
        stage.mkdir(mode=0o700)
        writers = {role: Writer(stage, role, counts[role], len(indices), guard) for role in ROLES}
        for s in sessions:
            session_export(s, grouped[(participant, s["session"])], subject, storage, writers, guard)
        roles = {role: writer.close() for role, writer in writers.items()}
        guard.reserve(mask_file.stat().st_size)
        shutil.copyfile(mask_file, stage / "native_voxel_indices.npy")
        publish_subject(stage, building, guard)
        summaries.append({"participant": participant, "roles": roles, "native_grid": subject["grid"],
                          "mask_index_sha256": subject["voxel_indices_sha256"], "column_order": "ascending_full_grid_C_order"})
        print(json.dumps({"participant": participant, "status": "regular_rows_provisioned_sealed", "contains_neural_values": False}), flush=True)
    require(len(summaries) == 5, "required_participants_missing")
    report = {"schema_version": "ep21.regular_neural_structural.v1", "publication": "complete", "job_id": guard.job,
              "contains_neural_values": False, "payload": str(destination), "subjects": summaries,
              "session_header_bindings": guard.bindings, "resources": RESOURCES,
              "durable_payload_bytes": guard.durable_bytes, "upstream_exclusions": "unverified_accepted_fixed_measurement",
              "OOD_arrays_formed_retained_summarized": False, "mixed_compressed_streams_traversed": True,
              "model_fitting_scoring_run": False, "resolver_neural_access_authorized": False}
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    guard.reserve(len(encoded), durable=True)
    building.chmod(0o500)
    building.rename(destination)
    temp_report = report_path.with_suffix(".json.pending")
    with temp_report.open("xb") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    temp_report.chmod(0o400)
    temp_report.rename(report_path)  # Completion marker only after all subjects.
    scratch.rmdir()
    print(json.dumps({"publication": "complete", "structural_report": str(report_path), "neural_fitting_scoring_run": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"publication": "incomplete", "error_type": type(exc).__name__,
                          "error_code": str(exc) if type(exc) is RuntimeError else "export_stopped"}), flush=True)
        raise SystemExit(1)
