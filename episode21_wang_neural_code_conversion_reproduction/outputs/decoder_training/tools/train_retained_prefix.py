"""Bounded EP21 retained real-data decoder prefix; no scheduler submission.

The exact approved authorization is required before scientific imports or payload
access. The existing lock and public conformance are reused, never regenerated.
This file has not been executed as part of its preparation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile


BASE = Path(__file__).resolve().parents[3]
PROV = BASE / "outputs/steward_provisioning"
OUTPUT = BASE / "outputs/decoder_training/retained_prefix_001"
AUTH = BASE / "outputs/decoder_training_prefix_authorization.json"
LOCK = BASE / "outputs/protocol_lock.yaml"
NEURAL = PROV / "regular_neural_sealed/sub-01"
FEATURES = PROV / "regular_vgg19_sealed/sub-01"
PRESERVED_FEATURES = PROV / "regular_vgg19_sealed.building.46424955/sub-01"
EARLY_AUTHORITY = "allow_completed_sub01_before_global_export"
SOURCE = BASE / "outputs/source_pipeline_binding/fastl2lir_0_9"
SOURCE_SHA = "2abd8f46c8faa69aaea4e56568c8c4048178c24cc1ea88ad7c33f661151a0412"
LAYERS = ("conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
          "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4",
          "conv5_1", "conv5_2", "conv5_3", "conv5_4", "fc6", "fc7", "fc8")
SHAPES = tuple([(64, 224, 224)] * 2 + [(128, 112, 112)] * 2
               + [(256, 56, 56)] * 4 + [(512, 28, 28)] * 4
               + [(512, 14, 14)] * 4 + [(4096,), (4096,), (1000,)])
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
RESOURCES = {"jobs": 1, "cpus": 1, "memory_gib": 7, "minutes": 120,
             "gpus": 0, "durable_bytes": 2_000_000_000,
             "scratch_bytes": 10_000_000_000}
MAX_SECONDS = 6900
PREFIX = 64
IMAGES = 4712
VOXELS = 500
ALPHA = 100
NEURAL_FILES = ("subject_unique_responses.npy", "subject_unique_trials.tsv",
                "subject_unique_finite_packed.npy", "native_voxel_indices.npy")
FEATURE_FILES = ("features.h5", "image_index.tsv", "finite_packed.h5")
AUTH_EXPECTED = {
    "schema_version": "ep21.decoder_prefix_authorization.v1", "episode_id": "ep21",
    "purpose": "retained_sub01_decoder_prefix_001", "approved": True,
    "participant": "sub-01", "training_role": "subject_unique", "training_images": IMAGES,
    "layers": list(LAYERS), "units_per_layer": PREFIX,
    "convolutional_units": "channel_0_spatial_Fortran_0_through_63",
    "FC_units": "indices_0_through_63", "resources": RESOURCES,
    "internal_seconds": MAX_SECONDS, "allow_neural_values": True,
    "allow_regular_feature_values": True, "allow_decoder_fitting": True,
    "allow_tau_train_values": False, "allow_tau_test_values": False,
    "allow_ood": False, "allow_prediction_or_scoring": False,
    "allow_converters": False, "allow_reconstruction": False,
    "allow_external_sites": False, "automatic_retry": False,
    "automatic_extension": False, "successor_authorized": False,
    "original_budget_consumption_preserved": True,
    "reuse_protocol_lock": "outputs/protocol_lock.yaml",
    "output": "outputs/decoder_training/retained_prefix_001",
    "neural_inputs": [str((NEURAL / n).relative_to(BASE)) for n in NEURAL_FILES],
    "feature_inputs": [str((FEATURES / n).relative_to(BASE)) for n in FEATURE_FILES],
}
np = None
h5py = None


class Stop(RuntimeError):
    pass


class UnitStop(Stop):
    def __init__(self, code, indices):
        super().__init__(code)
        self.indices = list(indices)


def require(ok, code):
    if not ok:
        raise Stop(code)


def read_json(path):
    return json.loads(Path(path).read_text())


def authorize(path):
    # Scope precedes the read: another CLI path cannot open a closed artifact.
    require(Path(path).absolute() == AUTH and not AUTH.is_symlink(), "fixed_authorization_path_required")
    record = read_json(AUTH)
    expected = dict(AUTH_EXPECTED)
    early_start = record.get(EARLY_AUTHORITY, False)
    require(early_start is True or early_start is False, "boolean_ordering_authority_required")
    if early_start:
        require(record.get(EARLY_AUTHORITY) is True
                and record.get("complete_feature_export_required") is False,
                "explicit_ordering_amendment_required")
        expected["feature_inputs"] = [str((PRESERVED_FEATURES / n).relative_to(BASE))
                                      for n in FEATURE_FILES]
    else:
        require(record.get(EARLY_AUTHORITY, False) is False
                and record.get("complete_feature_export_required") is True,
                "approved_complete_export_order_required")
    require(all(record.get(key) == value for key, value in expected.items()),
            "explicit_decoder_prefix_authorization_required")
    return record


def feature_source(*, early_start=False):
    # Pin the chosen physical directory for the entire attempt; never fall back
    # to whichever publication path happens to exist later. The continuation
    # hardlinks OLD_BUILDING/sub-01 into a different temporary tree and renames
    # only that NEW tree. Its authority preserves the original source directory.
    return PRESERVED_FEATURES if early_start else FEATURES


def admit_feature_export(lock, groups, *, early_start=False):
    source = feature_source(early_start=early_start)
    if early_start:
        original = read_json(BASE / "outputs/regular_vgg19_runtime_retry_authorization_2026-10-03.json")
        require(original.get("approved") is True
                and original.get("purpose") == "regular_vgg19_derivative_export",
                "original_feature_authority_required")
        claim = read_json(PROV / "regular_vgg19_export_job_claim.json")
        require(claim == {"job_id": "46424955", "purpose": original["purpose"],
                          "protocol_lock": str(LOCK)}, "original_feature_claim_required")
        history = PROV / "regular_vgg19_attempt003-46424955.out"
        records = [json.loads(line) for line in history.read_text().splitlines()
                   if line.startswith("{")]
        require(records and records[-1] == {"publication": "incomplete",
                "error_type": "RuntimeError", "error_code": "deadline_reached"},
                "original_feature_deadline_closeout_required")
        sealed = [row.get("partition") for row in records
                  if row.get("status") == "real_regular_features_provisioned_sealed"]
        require(sealed == ["sub-01", "sub-03"], "original_sub01_sealed_completion_required")
        require(source.resolve() == source and source.is_dir()
                and {p.name for p in source.iterdir()} == set(FEATURE_FILES)
                and all((source / name).is_file() and not (source / name).is_symlink()
                        for name in FEATURE_FILES), "preserved_sub01_three_file_source_required")
        return source, {"mode": "explicitly_amended_completed_sub01_only",
                        "physical_source": str(source), "original_job_id": "46424955",
                        "completion_evidence": str(history),
                        "global_feature_export_completion_claimed": False}
    report = read_json(PROV / "regular_vgg19_structural_report.json")
    require(report.get("publication") == "complete"
            and report.get("feature_recipe") == lock["feature_recipe"]
            and report.get("layer_shapes") == dict(zip(LAYERS, map(list, SHAPES)))
            and report.get("feature_dtype") == "float32"
            and report.get("OOD_access") is False, "complete_regular_feature_export_required")
    parts = report.get("partitions", [])
    require([row["partition"] for row in parts] == list(groups)
            and all(row["images"] == len(groups[row["partition"]]) and row["nonfinite_elements"] == 0
                    for row in parts), "complete_frozen_feature_partitions_required")
    return source, {"mode": "approved_complete_regular_export",
                    "physical_source": str(source),
                    "global_feature_export_completion_claimed": True}


def preserved_feature_headers(source, lock, check):
    """All 19 headers only, before neural values; no HDF5 dataset values."""
    import h5py as metadata_h5py
    require(metadata_h5py.__version__ == lock["runtime"]["h5py"],
            "locked_feature_header_runtime_required")
    with metadata_h5py.File(source / "features.h5", "r") as features, metadata_h5py.File(source / "finite_packed.h5", "r") as finite:
        require(set(features) == set(finite) == set(LAYERS), "complete_locked_layer_vocabulary_required")
        for layer, shape in zip(LAYERS, SHAPES):
            check()
            require(features[layer].shape == (IMAGES,) + shape and features[layer].dtype.str == "<f4"
                    and finite[layer].shape == (IMAGES, (math.prod(shape) + 7) // 8)
                    and finite[layer].dtype.str == "|u1", "preserved_feature_layer_header_mismatch")


def seconds(text):
    days, clock = text.split("-", 1) if "-" in text else ("0", text)
    parts = clock.split(":")
    require(len(parts) == 3, "finite_allocation_time_required")
    hours, minutes, secs = map(int, parts)
    return int(days) * 86400 + hours * 3600 + minutes * 60 + secs


def verify_allocation():
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and "-ln" not in host, "actual_compute_allocation_required")
    require(all(os.environ.get(name) == "1" for name in THREADS), "six_thread_caps_must_equal_one")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "7168", "one_CPU_7GiB_task_required")
    # One allocation admission lookup, not monitoring/polling or submission.
    result = subprocess.run(["scontrol", "show", "job", job, "-o"], check=True,
                            capture_output=True, text=True, timeout=30)
    fields = dict(token.split("=", 1) for token in result.stdout.split() if "=" in token)
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING"
            and fields.get("NumCPUs") == "1" and fields.get("NumNodes") == "1"
            and fields.get("NodeList") == host, "actual_single_CPU_node_binding_required")
    tres = dict(token.split("=", 1) for token in fields.get("AllocTRES", "").split(",") if "=" in token)
    require(tres.get("cpu") == "1" and tres.get("mem") in ("7G", "7168M")
            and not any(key.startswith("gres/gpu") for key in tres), "actual_allocation_resource_mismatch")
    limit = seconds(fields["TimeLimit"])
    require(0 < limit <= 7200, "two_hour_maximum_allocation_required")
    end = int(os.environ.get("SLURM_JOB_END_TIME", "0"))
    require(end > time.time() + 300, "allocation_closure_reserve_required")
    return {"job_id": job, "hostname": host, "allocated_cpus": 1,
            "memory_gib": 7, "gpus": 0, "time_limit_seconds": limit,
            "job_end_epoch": end}


class Budget:
    def __init__(self, allocation, max_seconds):
        self.started = time.monotonic()
        self.seconds = min(max_seconds, allocation["job_end_epoch"] - time.time() - 300)
        require(self.seconds > 0, "allocation_deadline_reached")
        self.deadline = self.started + self.seconds
        self.files = {}
        self.logical_read_bytes = 0

    def check(self):
        require(time.monotonic() < self.deadline, "prefix_deadline_reached")

    def room(self, extra):
        require(sum(self.files.values()) + extra <= RESOURCES["durable_bytes"], "durable_byte_cap")

    def note(self, path):
        self.files[str(path)] = Path(path).stat().st_size
        self.room(0)


def fsync_file(handle):
    handle.flush()
    os.fsync(handle.fileno())


def fsync_h5(handle):
    handle.flush()
    descriptor = handle.id.get_vfd_handle()
    require(isinstance(descriptor, int), "HDF5_fsync_descriptor_required")
    os.fsync(descriptor)


def publish_bytes(path, payload, budget, *, closing=False):
    require(not path.exists(), "existing_artifact_no_overwrite")
    if not closing:
        budget.check()
    budget.room(len(payload) + 4096)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            fsync_file(stream)
        os.link(temporary, path)
        budget.note(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_json(path, value, budget, *, closing=False):
    publish_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(),
                  budget, closing=closing)


def publish_arrays(path, arrays, budget):
    budget.check()
    budget.room(sum(value.nbytes for value in arrays.values()) + 1_048_576)
    require(not path.exists(), "existing_shared_state_no_overwrite")
    temporary = path.with_name("." + path.name + ".pending")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
            fsync_file(stream)
        os.link(temporary, path)
        budget.note(path)
    finally:
        temporary.unlink(missing_ok=True)


def unit_specs(layer, shape):
    specs = []
    for unit in range(PREFIX):
        if len(shape) == 3:
            channels, height, width = shape
            coordinate = [0, unit % height, unit // height]
            require(coordinate[2] < width, "prefix_outside_layer")
            flat_f = channels * unit
            flat_c = coordinate[1] * width + coordinate[2]
        else:
            coordinate, flat_f, flat_c = [unit], unit, unit
        specs.append({"layer": layer, "prefix_index": unit, "tensor_coordinate": coordinate,
                      "full_layer_Fortran_index": flat_f, "full_layer_C_index": flat_c,
                      "status": "pending"})
    return specs


def admit_metadata(*, early_start=False):
    sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
    import feature_protocol as fp
    require(tuple(fp.LAYERS) == LAYERS and tuple(map(tuple, fp.SHAPES)) == SHAPES,
            "locked_layer_order_and_shapes_required")
    lock = read_json(LOCK)  # The existing .yaml is a JSON-encoded immutable lock.
    complete = read_json(BASE / "outputs/pipeline_conformance/attempt-008/conformance_report.json")
    public = read_json(BASE / "outputs/pipeline_conformance/attempt-007/public_runtime_conformance_report.json")
    require(lock.get("state") == "valid_prospective_lock" and lock.get("job_id") == "46424955"
            and lock.get("neural_execution_authorized") is False
            and lock.get("OOD_access_authorized") is False, "existing_lock008_required")
    require(complete.get("outcome") == "passed_required_shared_and_Finding_A_protocol_conformance"
            and complete.get("job_id") == "46424955"
            and public.get("outcome") == "passed_public_VGG_strict_load_and_focused_decoder_analytic_fixture"
            and public.get("job_id") == "46424559", "reuse_completed_conformance007_008_required")
    require(all(lock["module_states"][key] == "runnable_locked" for key in
                ("laion_target_decoders", "laion_within_source_decoders", "finding_a_feature_decoding")),
            "native_decoders_not_locked_runnable")
    contract = lock["resolved_protocol_table"]["feature_decoder_contract"]
    require(contract["planned_penalty"] == ALPHA
            and contract["planned_voxel_selection"]["count_per_feature_unit"] == VOXELS
            and contract["planned_voxel_selection"]["ties_and_fewer_than_500_action"]
            == "ascending_stable_native_voxel_ID_ties_fewer_than_500_not_evaluable"
            and contract["response_and_feature_normalization_axes"]
            == "training_axis_0_means_and_standard_deviations_ddof_1_source_qualification_attempt_019"
            and contract["intercept_and_coefficient_convention"]
            == "authenticated_fastl2lir_0_9_sum_SSE_alpha_penalized_retained_bias_Fortran_target_unit_layout",
            "frozen_decoder_contract_mismatch")
    _, groups, alignment, source_metadata = fp.image_roles()
    require(alignment == lock["regular_image_binding"], "original_image_binding_required")
    if early_start:
        original = read_json(BASE / "outputs/regular_vgg19_runtime_retry_authorization_2026-10-03.json")
        fp.validate_feature_lock(lock, original, public, alignment)
    selected = groups["sub-01"]
    feature_root, feature_admission = admit_feature_export(lock, groups, early_start=early_start)
    neural_report = read_json(PROV / "regular_neural_structural_report.json")
    require(neural_report.get("publication") == "complete"
            and neural_report.get("contains_neural_values") is False
            and neural_report.get("OOD_arrays_formed_retained_summarized") is False,
            "completed_regular_neural_provisioning_required")
    subjects = [row for row in neural_report["subjects"] if row["participant"] == "sub-01"]
    require(len(subjects) == 1, "one_sub01_metadata_binding_required")
    subject = subjects[0]
    role = subject["roles"]["subject_unique"]
    require(role == {"rows": 18843, "voxels": 12295, "nonfinite_elements": 0,
                     "finite_mask_bitorder": "little", "dtype": "float32", "repetitions_preserved": True}
            and subject["column_order"] == "ascending_full_grid_C_order", "sub01_structural_support_changed")
    return lock, selected, subject, feature_root, feature_admission


def feature_index(path, selected):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["output_row_index", "regular_HDF5_row", "image_id", "participant", "role"],
                "feature_index_header_mismatch")
        ids = []
        for index, (ordinal, expected) in enumerate(selected):
            row = next(reader, None)
            require(row == {"output_row_index": str(index), "regular_HDF5_row": str(ordinal),
                            "image_id": expected["image_id"], "participant": "sub-01", "role": "subject_unique"},
                    "feature_index_role_or_identity_mismatch")
            ids.append(row["image_id"])
        require(next(reader, None) is None and len(ids) == IMAGES and len(set(ids)) == IMAGES,
                "complete_unique_feature_rows_required")
    return ids


def group_trials(rows, image_ids, expected_rows):
    groups = {image: [] for image in image_ids}
    identities = set()
    require(len(groups) == len(image_ids), "duplicate_image_identity")
    for index, row in enumerate(rows):
        require(row["output_row_index"] == str(index) and row["participant"] == "sub-01"
                and row["role"] == "subject_unique" and row["image_id"] in groups,
                "neural_trial_role_or_row_identity_mismatch")
        identity = (row["session"], row["beta_volume_index"])
        require(identity not in identities, "duplicate_neural_trial_identity")
        identities.add(identity)
        groups[row["image_id"]].append(index)
    require(len(rows) == expected_rows and all(groups.values()), "all_images_and_repetitions_required")
    return [groups[image] for image in image_ids]


def aggregate_images(values, flags, groups, check):
    result = np.empty((len(groups), values.shape[1]), dtype=np.float32)
    counts = np.empty(len(groups), dtype=np.int64)
    for image, indices in enumerate(groups):
        check()
        trials = values[indices, :]
        finite = np.unpackbits(flags[indices, :], axis=1, bitorder="little")[:, :values.shape[1]]
        require(bool(finite.all()) and bool(np.isfinite(trials).all()), "nonfinite_required_training_repetition")
        # LAION adaptation: equal image weighting after role-assigned averaging.
        # Preserve stored repetition order and NumPy float32 mean arithmetic.
        result[image] = np.mean(trials, axis=0)
        counts[image] = len(indices)
    require(bool(np.isfinite(result).all()), "nonfinite_required_image_mean")
    return result, counts


def normalize_neural(averaged, native_ids):
    require(averaged.dtype == np.float32 and bool(np.isfinite(averaged).all()), "finite_float32_image_means_required")
    means = np.mean(averaged, axis=0)[np.newaxis, :]
    scales = np.std(averaged, axis=0, ddof=1)[np.newaxis, :]
    require(bool(np.isfinite(means).all()) and bool(np.isfinite(scales).all()), "nonfinite_neural_training_moments")
    eligible = scales[0] > 0
    require(int(eligible.sum()) >= VOXELS, "fewer_than_500_eligible_voxels")
    # Caller normalization precedes the source fit float32 cast.
    normalized = ((averaged[:, eligible] - means[:, eligible]) / scales[:, eligible]).astype(np.float32)
    require(bool(np.isfinite(normalized).all()), "nonfinite_normalized_neural_training_input")
    return normalized, native_ids[eligible], means, scales, eligible


def source_corrmat():
    wheel = SOURCE / "fastl2lir-0.9-py2.py3-none-any.whl"
    require(hashlib.sha256(wheel.read_bytes()).hexdigest() == SOURCE_SHA, "pinned_decoder_package_required")
    code = (SOURCE / "plaintext/fastl2lir/fastl2lir.py").read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        require(code == archive.read("fastl2lir/fastl2lir.py"), "pinned_decoder_plaintext_required")
    tree = ast.parse(code)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "corrmat"]
    require(len(definitions) == 1, "one_qualified_corrmat_required")
    scope = {"np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE / "plaintext/fastl2lir/fastl2lir.py"), "exec"), scope)
    return scope["corrmat"]


def source_gram(x):
    augmented = np.hstack((x, np.ones((x.shape[0], 1), dtype=np.float32)))
    gram = np.matmul(augmented.T, augmented) + ALPHA * np.eye(augmented.shape[1], dtype=np.float32)
    require(bool(np.isfinite(gram).all()), "nonfinite_full_penalized_Gram")
    return augmented, gram


def prepare_layer_math(x, augmented, gram, targets, corrmat):
    means = np.mean(targets, axis=0)[np.newaxis, :]
    scales = np.std(targets, axis=0, ddof=1)[np.newaxis, :]
    bad = np.flatnonzero(~np.isfinite(means[0]) | ~np.isfinite(scales[0]) | (scales[0] <= 0))
    if bad.size:
        return {"bad_units": bad.tolist(), "error": "undefined_required_feature_training_scale"}
    normalized = ((targets - means) / scales).astype(np.float32)
    require(bool(np.isfinite(normalized).all()), "nonfinite_normalized_feature_training_input")
    if x.shape[1] == VOXELS:
        # Exact pinned-source dispatch: no selection, native order, X.T @ Y.
        right = np.matmul(augmented.T, normalized)
        solved = np.linalg.solve(gram, right)
        require(bool(np.isfinite(solved).all()), "nonfinite_all_feature_source_solve")
        return {"means": means, "scales": scales, "solved": solved,
                "branch": "source_exactly_500_native_order_no_selection"}
    low_variance = np.nonzero(np.var(x, axis=0) < 0.00000001)
    correlations = corrmat(x, normalized, "col")
    correlations[low_variance, :] = 0.0
    require(bool(np.isfinite(correlations).all()), "nonfinite_required_training_Pearson")
    right = np.matmul(normalized.T, augmented)
    require(bool(np.isfinite(right).all()), "nonfinite_source_right_hand_side")
    return {"means": means, "scales": scales, "correlations": correlations.T,
            "right": right, "branch": "source_greater_than_500_with_native_ID_ties"}


def solve_selected(gram, right, correlations, native_ids):
    require(len(native_ids) > VOXELS and bool(np.isfinite(correlations).all()), "selected_branch_support_required")
    selected = np.lexsort((native_ids, -abs(correlations)))[:VOXELS]
    augmented_ids = np.hstack((selected, len(native_ids)))  # bias last, source rank order retained
    submatrix = (gram.ravel()[(augmented_ids + (augmented_ids * gram.shape[1]).reshape((-1, 1))).ravel()]).reshape(501, 501)
    solved = np.linalg.solve(submatrix, right[augmented_ids].reshape(-1, 1))
    require(bool(np.isfinite(solved).all()), "nonfinite_selected_source_solve")
    return selected, solved[:VOXELS, 0], solved[-1, 0]


def read_layer(features, finite, layer, shape, specs, budget):
    data, flags = features[layer], finite[layer]
    require(data.shape == (IMAGES,) + shape and data.dtype.str == "<f4"
            and flags.shape == (IMAGES, (int(np.prod(shape)) + 7) // 8)
            and flags.dtype.str == "|u1", "selected_feature_layer_header_mismatch")
    layout = {"shape": list(data.shape), "dtype": str(data.dtype),
              "chunks": list(data.chunks) if data.chunks else None,
              "compression": data.compression, "compression_options": data.compression_opts}
    columns = np.asarray([spec["full_layer_C_index"] for spec in specs], dtype=np.int64)
    byte_columns, inverse = np.unique(columns // 8, return_inverse=True)
    target = np.empty((IMAGES, PREFIX), dtype=np.float32)
    for image in range(IMAGES):
        budget.check()
        if len(shape) == 3:
            plane = data[image, 0, :, :]
            target[image] = plane.reshape(-1, order="F")[:PREFIX]
        else:
            plane = data[image, :]
            target[image] = plane[:PREFIX]
        packed = flags[image, byte_columns]
        bits = (packed[inverse] >> (columns % 8)) & 1
        bad = np.flatnonzero((bits == 0) | ~np.isfinite(target[image]))
        if bad.size:
            raise UnitStop("nonfinite_required_prefix_target", bad.tolist())
        budget.logical_read_bytes += plane.nbytes + packed.nbytes
    return target, layout


class LayerStore:
    """Fresh per-layer container: completed rows are immutable, pending rows explicit."""
    def __init__(self, path, specs, budget):
        budget.check()
        budget.room(1_048_576)
        self.path, self.specs, self.budget = path, specs, budget
        self.file = h5py.File(path, "x")
        out = self.file
        out.attrs["schema_version"] = "ep21.retained_decoder_prefix_layer.v1"
        out.attrs["layer"] = specs[0]["layer"]
        out.attrs["complete_decoder"] = False
        out.attrs["source_fragment_axes"] = "unit_by_ranked_selected_native_voxel; bias_separate"
        out.attrs["state_codes"] = "0=pending,1=in_progress,2=complete,3=failed,4=interrupted"
        out.create_dataset("state", (PREFIX,), dtype="u1", fillvalue=0)
        out.create_dataset("selected_native_voxel_ids", (PREFIX, VOXELS), dtype="<i8", fillvalue=-1)
        out.create_dataset("weights", (PREFIX, VOXELS), dtype="<f4", fillvalue=np.nan)
        for name in ("bias", "target_mean", "target_std"):
            out.create_dataset(name, (PREFIX,), dtype="<f4", fillvalue=np.nan)
        out.create_dataset("solve_seconds", (PREFIX,), dtype="<f8", fillvalue=np.nan)
        out.create_dataset("full_layer_Fortran_index", data=[spec["full_layer_Fortran_index"] for spec in specs], dtype="<i8")
        out.create_dataset("tensor_coordinate", data=[spec["tensor_coordinate"] for spec in specs], dtype="<i8")
        fsync_h5(out)
        budget.note(path)

    def start(self, index):
        self.budget.check()
        require(int(self.file["state"][index]) == 0, "unit_already_started_no_replay")
        self.file["state"][index] = 1
        fsync_h5(self.file)
        self.specs[index]["status"] = "in_progress"

    def complete(self, index, native_ids, weights, bias, mean, std, elapsed):
        require(int(self.file["state"][index]) == 1, "unit_not_owned_in_progress")
        require(native_ids.shape == weights.shape == (VOXELS,)
                and bool(np.isfinite(weights).all()) and bool(np.isfinite([bias, mean, std, elapsed]).all())
                and std > 0, "valid_complete_unit_required")
        # Defer the internal deadline only across this small publication
        # transaction, so a committed marker and the in-memory status agree.
        # The allocation's hard stop is never extended.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            self.file["selected_native_voxel_ids"][index] = native_ids
            self.file["weights"][index] = weights
            self.file["bias"][index] = bias
            self.file["target_mean"][index] = mean
            self.file["target_std"][index] = std
            self.file["solve_seconds"][index] = elapsed
            fsync_h5(self.file)  # data durable before the completion marker
            self.file["state"][index] = 2
            fsync_h5(self.file)
            self.specs[index]["status"] = "complete"
            self.budget.note(self.path)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def fail(self, indices, error, interrupted=False):
        for index in indices:
            require(int(self.file["state"][index]) != 2, "cannot_change_completed_unit")
            self.file["state"][index] = 4 if interrupted else 3
            self.specs[index].update(status="interrupted" if interrupted else "failed", error=error)
        fsync_h5(self.file)
        self.budget.note(self.path)

    def close(self):
        if self.file:
            self.file.close()


def load_libraries():
    global np, h5py
    import numpy
    import h5py as imported_h5py
    np, h5py = numpy, imported_h5py


def run(args):
    require(Path.cwd().resolve() == BASE, "exact_episode_working_directory_required")
    require(args.participant == "sub-01" and args.units_per_layer == PREFIX and args.workers == 1
            and 0 < args.max_seconds <= MAX_SECONDS, "fixed_prefix_and_worker_required")
    authorization = authorize(args.authorization)
    early_start = authorization.get(EARLY_AUTHORITY, False)
    allocation = verify_allocation()
    if early_start:
        require(allocation["job_id"] == "46687670", "existing_prefix_job_only")
    require(not OUTPUT.exists(), "existing_attempt_no_replay_or_overwrite")
    budget = Budget(allocation, args.max_seconds)
    lock, selected, subject, feature_root, feature_admission = admit_metadata(early_start=early_start)
    budget.check()
    ids = feature_index(feature_root / "image_index.tsv", selected)
    if early_start:
        preserved_feature_headers(feature_root, lock, budget.check)
    with (NEURAL / "subject_unique_trials.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["output_row_index", "participant", "session", "run", "beta_index",
                                     "beta_volume_index", "image_id", "role"], "neural_trial_index_header_mismatch")
        trials = list(reader)
    groups = group_trials(trials, ids, subject["roles"]["subject_unique"]["rows"])
    # Admission is complete before any attempt directory or scientific payload.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(mode=0o700)
    scratch = Path(os.environ["SCRATCH"]) / "br_autoresearch" / BASE.name / ("decoder-prefix-" + allocation["job_id"])
    scratch.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(scratch)
    all_specs = {layer: unit_specs(layer, shape) for layer, shape in zip(LAYERS, SHAPES)}
    record = {"schema_version": "ep21.retained_decoder_prefix.v1", "analysis_id": "laion_decoder_training",
              "status": "admitted_pending", "authorization": str(AUTH), "allocation": allocation,
              "resources": RESOURCES, "max_work_seconds": budget.seconds,
              "participant": "sub-01", "training_role": "subject_unique", "training_images": IMAGES,
              "training_repetitions": len(trials), "units_requested": PREFIX * len(LAYERS),
              "units": all_specs, "completed_units": 0, "neural_values_read": False,
              "regular_feature_values_read": False, "predictions_or_scores_computed": False,
              "tau_train_values_read": False, "tau_test_values_read": False, "OOD_access": False,
              "complete_decoder": False, "successor_authorized": False,
              "feature_admission": feature_admission,
              "normalization_and_averaging": "role_assigned_float32_image_means_then_training_mean_sample_std_before_float32_fit",
              "shared_state": "shared_training_state.npz", "layer_timings": {}}
    publish_json(OUTPUT / "requested_units.json", record, budget)
    active_store = None
    current = None
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise Stop("prefix_deadline_reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, budget.deadline - time.monotonic()))
    error = None
    try:
        load_libraries()
        runtime = {"python": sys.version.split()[0], "numpy": np.__version__, "h5py": h5py.__version__,
                   "interpreter": sys.executable, "prefix": sys.prefix,
                   "loaded_modules": os.environ.get("LOADEDMODULES", "")}
        require(all(runtime[key] == lock["runtime"][key] for key in ("python", "numpy", "h5py")),
                "locked_decoder_runtime_mismatch")
        record["runtime"] = runtime
        record["thread_control"] = {"pre_import_environment": {name: os.environ[name] for name in THREADS},
                                    "workers": 1, "verified_allocated_CPUs": 1}
        corrmat = source_corrmat()
        start = time.monotonic()
        native_path = NEURAL / "native_voxel_indices.npy"
        require(hashlib.sha256(native_path.read_bytes()).hexdigest() == subject["mask_index_sha256"], "frozen_native_ID_binding_mismatch")
        native = np.load(native_path, allow_pickle=False)
        require(native.shape == (12295,) and native.dtype.kind in "iu"
                and bool((native[1:] > native[:-1]).all()), "canonical_native_ID_order_required")
        record["neural_values_read"] = True
        values = np.load(NEURAL / "subject_unique_responses.npy", mmap_mode="r", allow_pickle=False)
        flags = np.load(NEURAL / "subject_unique_finite_packed.npy", mmap_mode="r", allow_pickle=False)
        require(values.shape == (18843, 12295) and values.dtype.str == "<f4"
                and flags.shape == (18843, (12295 + 7) // 8) and flags.dtype.str == "|u1", "neural_role_payload_shape_mismatch")
        averaged, repetitions = aggregate_images(values, flags, groups, budget.check)
        budget.logical_read_bytes += values.nbytes + flags.nbytes
        del values, flags
        normalized, eligible_ids, means, scales, eligible = normalize_neural(averaged, native)
        del averaged
        record["aggregation_normalization_seconds"] = time.monotonic() - start
        record["eligible_voxels"] = len(eligible_ids)
        start = time.monotonic()
        augmented, gram = source_gram(normalized)
        budget.check()
        record["shared_Gram_seconds"] = time.monotonic() - start
        publish_arrays(OUTPUT / "shared_training_state.npz", {
            "normalized_X": normalized, "penalized_Gram": gram, "all_native_voxel_ids": native,
            "eligible_native_voxel_ids": eligible_ids, "eligible_mask": eligible,
            "neural_mean": means, "neural_std": scales, "image_repetition_counts": repetitions,
            "training_image_ids": np.asarray(ids)}, budget)
        publish_json(OUTPUT / "training_row_binding.json", {"training_image_ids": ids,
                     "preserved_trial_rows_per_image": groups, "role": "subject_unique",
                     "participant": "sub-01", "image_mean_weighting": "equal_images"}, budget)
        record["regular_feature_values_read"] = True
        with h5py.File(feature_root / "features.h5", "r") as features, h5py.File(feature_root / "finite_packed.h5", "r") as finite:
            require(set(features) == set(finite) == set(LAYERS), "complete_locked_layer_vocabulary_required")
            for layer, shape in zip(LAYERS, SHAPES):
                budget.check()
                specs = all_specs[layer]
                active_store = LayerStore(OUTPUT / (layer + ".h5"), specs, budget)
                timing = record["layer_timings"][layer] = {}
                start = time.monotonic()
                try:
                    targets, layout = read_layer(features, finite, layer, shape, specs, budget)
                except UnitStop as exc:
                    active_store.fail(exc.indices, str(exc))
                    raise
                timing["target_loading_seconds"] = time.monotonic() - start
                timing["source_layout"] = layout
                start = time.monotonic()
                math = prepare_layer_math(normalized, augmented, gram, targets, corrmat)
                del targets
                timing["normalization_ranking_setup_seconds"] = time.monotonic() - start
                if "bad_units" in math:
                    active_store.fail(math["bad_units"], math["error"])
                    raise Stop(math["error"])
                timing["source_branch"] = math["branch"]
                timing["solve_seconds"] = 0.0
                timing["publication_seconds"] = 0.0
                for index in range(PREFIX):
                    budget.check()
                    current = index
                    active_store.start(index)
                    start = time.monotonic()
                    if "solved" in math:
                        indices = np.arange(VOXELS)
                        weights, bias = math["solved"][:VOXELS, index], math["solved"][-1, index]
                    else:
                        indices, weights, bias = solve_selected(gram, math["right"][index], math["correlations"][index], eligible_ids)
                    elapsed = time.monotonic() - start
                    timing["solve_seconds"] += elapsed
                    start = time.monotonic()
                    active_store.complete(index, eligible_ids[indices], weights, bias,
                                          math["means"][0, index], math["scales"][0, index], elapsed)
                    timing["publication_seconds"] += time.monotonic() - start
                    record["completed_units"] += 1
                    current = None
                active_store.close()
                active_store = None
                del math
        record["status"] = "complete_retained_prefix_only"
        require(record["completed_units"] == PREFIX * len(LAYERS), "full_fixed_prefix_accounting_required")
    except Exception as exc:
        error = exc
        record["status"] = "partial_prefix_stopped"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if active_store is not None:
            try:
                if current is not None and all_specs[active_store.specs[0]["layer"]][current]["status"] != "complete":
                    active_store.fail([current], record.get("error", "interrupted"),
                                      interrupted=record.get("error") == "prefix_deadline_reached")
            except Exception as close_error:
                record["failure_marker_error"] = type(close_error).__name__
            finally:
                active_store.close()
        record["elapsed_seconds"] = time.monotonic() - budget.started
        record["completed_units"] = sum(spec["status"] == "complete" for specs in all_specs.values() for spec in specs)
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["logical_input_bytes"] = budget.logical_read_bytes
        record["durable_bytes_before_final_status"] = sum(budget.files.values())
        record["scratch_bytes_written_by_worker"] = 0
        record["accounting"] = {"charge_parent_allocation_once": True, "child_step_additional_charge": False,
                                "prior_consumption_preserved": True, "maximum_CPU_hours": 2,
                                "actual_allocated_CPU_seconds_requires_coordinator_terminal_accounting": True}
        # Even abnormal process termination leaves requested_units plus flushed
        # per-layer states. This final manifest is once-only when closure succeeds.
        publish_json(OUTPUT / "prefix_status.json", record, budget, closing=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--units-per-layer", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "decoder_prefix_not_completed", "error_type": type(exc).__name__,
                          "error": str(exc), "no_retry_or_successor": True}), flush=True)
        return 2
    print(json.dumps({"status": "retained_prefix_complete" if result == 0 else "retained_prefix_stopped",
                      "report": str(OUTPUT / "prefix_status.json"), "complete_decoder": False,
                      "no_retry_or_successor": True}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
