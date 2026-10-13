"""Unactivated, participant-specific EP21 decoder training state.

No submission, automatic build-on-read, retry, feature-value read, tau access,
prediction or scoring. Four new states are prepared sequentially in ONE future
authorized parent with one shared budget; sub01's completed state is immutable.
The generalized fitting worker owns scheduler admission and its runtime budget.
These entrypoints independently require the future full-stage authority and an
actual one-CPU compute shell before imports or empirical state access.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import zipfile

import train_retained_prefix as prefix
import train_channel_continuation as retained

BASE = prefix.BASE
FULL_AUTH = BASE / "outputs/decoder_training_full_authorization.json"
FULL_ROOT = BASE / "outputs/decoder_training/full_continuation_001"
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
NEW_PARTICIPANTS = PARTICIPANTS[1:]
# Authenticated regular_neural_structural_report.json, subject_unique only.
SUPPORT = {"sub-01": (18843, 12295), "sub-03": (18844, 12897),
           "sub-05": (18843, 10891), "sub-06": (18844, 12023), "sub-07": (18842, 12057)}
ARRAY_NAMES = ("normalized_X", "penalized_Gram", "all_native_voxel_ids", "eligible_native_voxel_ids",
               "eligible_mask", "neural_mean", "neural_std", "image_repetition_counts", "training_image_ids")
FULL_AUTH_EXPECTED = {
    "schema_version": "ep21.full_decoder_authorization.v1", "episode_id": "ep21",
    "purpose": "full_five_participant_decoder_training", "approved": True,
    "participants": list(PARTICIPANTS), "training_role": "subject_unique", "training_images": 4712,
    "numerical_block_units": 64,
    "reuse_protocol_lock": "outputs/protocol_lock.yaml",
    "work_manifest": "outputs/decoder_training/full_continuation_001/work_manifest.json",
    "allow_retained_neural_state": True,
    "allow_original_neural_values_for_state_build": list(NEW_PARTICIPANTS),
    "allow_regular_feature_values": True, "allow_decoder_fitting": True,
    "allow_decoder_fragment_assembly": True,
    "assembled_manifest": "outputs/decoder_training/full_continuation_001/assembled_decoder_manifest.json",
    "allow_tau_train_values": False, "allow_tau_test_values": False, "allow_ood": False,
    "allow_prediction_or_scoring": False, "allow_converters": False,
    "allow_reconstruction": False, "allow_external_sites": False,
    "automatic_retry": False, "automatic_extension": False, "successor_authorized": False,
    "original_budget_consumption_preserved": True,
}
require, Stop = prefix.require, prefix.Stop


def valid_job_resources(resources):
    """Finite future caps; this validates a proposal shape, never approves it."""
    require(isinstance(resources, dict)
            and all(type(resources.get(key)) is int for key in
                    ("cpus", "memory_gib", "gpus", "minutes", "durable_bytes", "scratch_bytes"))
            and resources["cpus"] == 1 and resources["memory_gib"] == 7 and resources["gpus"] == 0
            and 0 < resources["minutes"] <= 1440
            and resources["durable_bytes"] > 0 and resources["scratch_bytes"] > 0,
            "explicit_finite_one_CPU_7GiB_job_resources_required")
    return resources


def authorize_full(path):
    require(Path(path).absolute() == FULL_AUTH and not FULL_AUTH.is_symlink(),
            "separate_fixed_full_stage_authorization_required")
    record = prefix.read_json(FULL_AUTH)
    require(all(json.dumps(record.get(key), sort_keys=True) == json.dumps(value, sort_keys=True)
                for key, value in FULL_AUTH_EXPECTED.items()), "explicit_full_stage_authorization_required")
    # The fixed prefix/channel0 contribute 51,328. A later approved tranche
    # can increase this count; the shared manifest validator binds its exact
    # value to validated terminal coverage before any full-stage execution.
    require(type(record.get("reuse_completed_units")) is int
            and 51328 <= record["reuse_completed_units"] <= 74306440,
            "actual_retained_unit_count_required")
    caps = record.get("resources", {})
    keys = ("worker_jobs", "state_build_jobs", "assembly_jobs", "maximum_concurrent_jobs",
            "maximum_allocated_CPU_seconds", "durable_bytes", "peak_transient_bytes")
    require(isinstance(caps, dict) and all(type(caps.get(key)) is int and caps[key] > 0 for key in keys)
            and caps["state_build_jobs"] == 1 and caps["assembly_jobs"] == 1
            and caps["maximum_concurrent_jobs"] <= caps["worker_jobs"] + 2,
            "finite_aggregate_full_stage_resources_required")
    state_cap = valid_job_resources(record.get("state_build_resources"))
    assembly_cap = valid_job_resources(record.get("assembly_resources"))
    if "cpu_budget_mode" in record:
        import decoder_batch_accounting as accounting
        accounting.budget_definition(record)
    require((state_cap["minutes"] + assembly_cap["minutes"]) * 60 <= caps["maximum_allocated_CPU_seconds"]
            and state_cap["durable_bytes"] + assembly_cap["durable_bytes"] <= caps["durable_bytes"]
            and max(state_cap["scratch_bytes"], assembly_cap["scratch_bytes"]) <= caps["peak_transient_bytes"],
            "state_and_assembly_reservations_exceed_full_stage_caps")
    return record


def require_compute(budget):
    """The caller owns scheduler verification; do not duplicate status polling."""
    host = socket.gethostname().split(".")[0]
    require("-ln" not in host and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "7168"
            and all(os.environ.get(key) == "1" for key in prefix.THREADS),
            "actual_one_CPU_7GiB_compute_context_required")
    allocation = getattr(budget, "allocation", None)
    if isinstance(allocation, dict):
        require(allocation.get("job_id") == os.environ["SLURM_JOB_ID"]
                and allocation.get("hostname", "").split(".")[0] == host,
                "caller_verified_allocation_must_match_actual_shell")
        end = allocation["job_end_epoch"]
    else:
        end = int(os.environ.get("SLURM_JOB_END_TIME", "0"))
    remaining = end - time.time() - 300
    require(remaining > 0 and 0 < budget.deadline - time.monotonic() <= remaining + 1,
            "caller_budget_must_preserve_actual_end_reserve")
    budget.check()


def state_directory(participant):
    require(participant in PARTICIPANTS, "locked_participant_required")
    return prefix.OUTPUT if participant == "sub-01" else FULL_ROOT / "shared" / participant


@dataclass(frozen=True)
class StateBinding:
    participant: str
    directory: Path
    feature_root: Path
    image_ids: tuple
    repetition_counts: tuple
    training_repetitions: int
    native_count: int
    eligible_count: int

    @property
    def state_path(self):
        return self.directory / "shared_training_state.npz"

    @property
    def training_row_binding_path(self):
        return self.directory / "training_row_binding.json"


def feature_index(path, participant, selected):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["output_row_index", "regular_HDF5_row", "image_id", "participant", "role"],
                "feature_index_header_mismatch")
        ids = []
        for index, (ordinal, expected) in enumerate(selected):
            row = next(reader, None)
            require(row == {"output_row_index": str(index), "regular_HDF5_row": str(ordinal),
                            "image_id": expected["image_id"], "participant": participant, "role": "subject_unique"},
                    "participant_feature_index_identity_mismatch")
            ids.append(row["image_id"])
        require(next(reader, None) is None and len(ids) == prefix.IMAGES and len(set(ids)) == prefix.IMAGES,
                "all_4712_unique_training_images_required")
    return ids


def group_trials(rows, image_ids, participant, expected_rows):
    require(participant in PARTICIPANTS, "locked_participant_required")
    groups = {image: [] for image in image_ids}
    require(len(groups) == len(image_ids), "duplicate_training_image_identity")
    identities = set()
    for index, row in enumerate(rows):
        require(row["output_row_index"] == str(index) and row["participant"] == participant
                and row["role"] == "subject_unique" and row["image_id"] in groups,
                "participant_trial_role_or_identity_mismatch")
        identity = row["session"], row["beta_volume_index"]
        require(identity not in identities, "duplicate_native_trial_identity")
        identities.add(identity)
        groups[row["image_id"]].append(index)
    require(len(rows) == expected_rows and all(groups.values()), "all_training_images_and_repetitions_required")
    return [groups[image] for image in image_ids]


def admit_participant(participant):
    """Authenticated metadata only; the public payload entrypoints gate authority."""
    require(participant in PARTICIPANTS, "locked_participant_required")
    sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
    import feature_protocol as fp
    require(tuple(fp.LAYERS) == prefix.LAYERS and tuple(map(tuple, fp.SHAPES)) == prefix.SHAPES,
            "locked_layer_shapes_required")
    lock = prefix.read_json(prefix.LOCK)
    completed = prefix.read_json(BASE / "outputs/pipeline_conformance/attempt-008/conformance_report.json")
    public = prefix.read_json(BASE / "outputs/pipeline_conformance/attempt-007/public_runtime_conformance_report.json")
    require(lock.get("state") == "valid_prospective_lock" and lock.get("job_id") == "46424955"
            and lock.get("neural_execution_authorized") is False and lock.get("OOD_access_authorized") is False,
            "existing_lock008_required")
    require(completed.get("outcome") == "passed_required_shared_and_Finding_A_protocol_conformance"
            and completed.get("job_id") == "46424955"
            and public.get("outcome") == "passed_public_VGG_strict_load_and_focused_decoder_analytic_fixture"
            and public.get("job_id") == "46424559", "reuse_conformance007_008_required")
    require(all(lock["module_states"][key] == "runnable_locked" for key in
                ("laion_target_decoders", "laion_within_source_decoders", "finding_a_feature_decoding")),
            "native_decoder_modules_not_runnable")
    contract = lock["resolved_protocol_table"]["feature_decoder_contract"]
    require(contract["planned_penalty"] == 100
            and contract["planned_voxel_selection"]["count_per_feature_unit"] == 500
            and contract["planned_voxel_selection"]["ties_and_fewer_than_500_action"]
            == "ascending_stable_native_voxel_ID_ties_fewer_than_500_not_evaluable"
            and contract["response_and_feature_normalization_axes"]
            == "training_axis_0_means_and_standard_deviations_ddof_1_source_qualification_attempt_019"
            and contract["intercept_and_coefficient_convention"]
            == "authenticated_fastl2lir_0_9_sum_SSE_alpha_penalized_retained_bias_Fortran_target_unit_layout",
            "unchanged_source_decoder_contract_required")
    _, roles, alignment, _ = fp.image_roles()
    require(alignment == lock["regular_image_binding"], "unchanged_training_image_roles_required")
    # This existing helper admits the complete seven-partition export using
    # structural metadata only. Its sub01 default path is deliberately ignored.
    prefix.admit_feature_export(lock, roles)
    feature_root = prefix.PROV / "regular_vgg19_sealed" / participant
    neural_root = prefix.PROV / "regular_neural_sealed" / participant
    report = prefix.read_json(prefix.PROV / "regular_neural_structural_report.json")
    require(report.get("publication") == "complete" and report.get("contains_neural_values") is False
            and report.get("OOD_arrays_formed_retained_summarized") is False,
            "complete_regular_neural_export_required")
    matching = [row for row in report["subjects"] if row["participant"] == participant]
    require(len(matching) == 1, "one_participant_structural_binding_required")
    subject = matching[0]
    rows, native_count = SUPPORT[participant]
    require(subject["roles"]["subject_unique"] == {
        "rows": rows, "voxels": native_count, "nonfinite_elements": 0, "finite_mask_bitorder": "little",
        "dtype": "float32", "repetitions_preserved": True}
        and subject["column_order"] == "ascending_full_grid_C_order", "participant_support_changed")
    ids = feature_index(feature_root / "image_index.tsv", participant, roles[participant])
    with (neural_root / "subject_unique_trials.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["output_row_index", "participant", "session", "run", "beta_index",
                                     "beta_volume_index", "image_id", "role"], "neural_trial_header_mismatch")
        groups = group_trials(list(reader), ids, participant, rows)
    return {"participant": participant, "lock": lock, "feature_root": feature_root,
            "neural_root": neural_root, "ids": ids, "groups": groups, "rows": rows,
            "native_count": native_count, "mask_index_sha256": subject["mask_index_sha256"]}


def build_arrays(values, flags, native, groups, check):
    """Reuse the previously exercised mathematical implementation unchanged."""
    averaged, repetitions = prefix.aggregate_images(values, flags, groups, check)
    normalized, eligible_ids, means, scales, eligible = prefix.normalize_neural(averaged, native)
    del averaged
    augmented, gram = prefix.source_gram(normalized)
    del augmented
    check()
    return {"normalized_X": normalized, "penalized_Gram": gram, "all_native_voxel_ids": native,
            "eligible_native_voxel_ids": eligible_ids, "eligible_mask": eligible,
            "neural_mean": means, "neural_std": scales, "image_repetition_counts": repetitions}


def _runtime(lock):
    prefix.load_libraries()
    observed = {"python": sys.version.split()[0], "numpy": prefix.np.__version__, "h5py": prefix.h5py.__version__}
    require(all(observed[key] == lock["runtime"][key] for key in observed), "locked_state_runtime_required")
    return observed


def _binding(meta, eligible_count):
    return StateBinding(meta["participant"], state_directory(meta["participant"]), meta["feature_root"],
                        tuple(meta["ids"]), tuple(map(len, meta["groups"])), meta["rows"],
                        meta["native_count"], eligible_count)


def prepare_state(participant, authorization_path, budget):
    """Build one new state; call sequentially for the four participants in one job.

    No output may preexist, even after an earlier failed attempt. Any repair or
    rerun needs a newly reviewed assignment, never deletion/reset of that state.
    """
    auth = authorize_full(authorization_path)
    require(participant in NEW_PARTICIPANTS, "sub01_state_is_retained_never_rebuilt")
    require_compute(budget)
    cap = auth["state_build_resources"]
    require(0 < budget.seconds <= cap["minutes"] * 60, "whole_state_parent_time_cap_required")
    meta = admit_participant(participant)
    # Bound a single new state's arrays and small row/status records. The cap is
    # shared by all four sequential calls; never reserve its full allowance anew.
    native_count = meta["native_count"]
    estimated_bytes = 4 * prefix.IMAGES * native_count + 4 * (native_count + 1) ** 2 + 8 * 1024**2
    require(sum(budget.files.values()) + estimated_bytes <= cap["durable_bytes"],
            "whole_state_parent_durable_cap_required")
    budget.room(estimated_bytes)
    directory = state_directory(participant)
    require(not directory.exists() and not directory.is_symlink(), "existing_state_no_overwrite_or_retry")
    runtime = _runtime(meta["lock"])
    budget.check()
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700)
    np = prefix.np
    record = {"schema_version": "ep21.participant_training_state.v1", "status": "state_build_started",
              "participant": participant, "training_role": "subject_unique", "training_images": 4712,
              "training_repetitions": meta["rows"], "native_voxels": meta["native_count"],
              "native_index_source_sha256": meta["mask_index_sha256"], "runtime": runtime,
              "source_job_id": os.environ["SLURM_JOB_ID"], "authorization": str(FULL_AUTH),
              "shared_state": "shared_training_state.npz", "training_row_binding": "training_row_binding.json",
              "original_neural_values_read": False, "regular_feature_values_read": False,
              "tau_train_values_read": False, "tau_test_values_read": False, "OOD_access": False,
              "predictions_or_scores_computed": False, "complete_decoder": False,
              "original_budget_consumption_preserved": True, "automatic_retry": False}
    prefix.publish_json(directory / "state_request.json", record, budget)
    values = flags = None
    error = None
    started = time.monotonic()
    try:
        native_path = meta["neural_root"] / "native_voxel_indices.npy"
        require(hashlib.sha256(native_path.read_bytes()).hexdigest() == meta["mask_index_sha256"],
                "participant_native_index_source_binding_mismatch")
        native = np.load(native_path, allow_pickle=False)
        require(native.shape == (meta["native_count"],) and native.dtype.kind in "iu"
                and bool((native[1:] > native[:-1]).all()), "ascending_participant_native_IDs_required")
        record["original_neural_values_read"] = True
        values = np.load(meta["neural_root"] / "subject_unique_responses.npy", mmap_mode="r", allow_pickle=False)
        flags = np.load(meta["neural_root"] / "subject_unique_finite_packed.npy", mmap_mode="r", allow_pickle=False)
        require(values.shape == (meta["rows"], meta["native_count"]) and values.dtype.str == "<f4"
                and flags.shape == (meta["rows"], (meta["native_count"] + 7) // 8)
                and flags.dtype.str == "|u1", "participant_subject_unique_payload_headers_required")
        arrays = build_arrays(values, flags, native, meta["groups"], budget.check)
        budget.logical_read_bytes += values.nbytes + flags.nbytes
        arrays["training_image_ids"] = np.asarray(meta["ids"])
        record["eligible_voxels"] = len(arrays["eligible_native_voxel_ids"])
        prefix.publish_arrays(directory / "shared_training_state.npz", arrays, budget)
        prefix.publish_json(directory / "training_row_binding.json", {
            "training_image_ids": meta["ids"], "preserved_trial_rows_per_image": meta["groups"],
            "role": "subject_unique", "participant": participant, "image_mean_weighting": "equal_images"}, budget)
        record["status"] = "complete_participant_training_state_only"
        del arrays
    except Exception as exc:
        error = exc
        record.update(status="participant_training_state_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        record["cleanup_errors"] = []
        for name, array in (("neural_response_mmap", values), ("finite_flag_mmap", flags)):
            if array is not None:
                try:
                    array._mmap.close()
                except Exception as exc:
                    record["cleanup_errors"].append({"object": name, "error_type": type(exc).__name__, "error": str(exc)})
        record["elapsed_seconds"] = time.monotonic() - started
        prefix.publish_json(directory / "state_status.json", record, budget, closing=True)
    if error is not None:
        raise error
    return _binding(meta, record["eligible_voxels"])


def _admit_state_binding(meta):
    directory = state_directory(meta["participant"])
    if meta["participant"] == "sub-01":
        require(retained.admit_prefix(meta["ids"]) == meta["groups"], "retained_sub01_repetition_binding_changed")
        eligible_count = retained.ELIGIBLE
    else:
        record = prefix.read_json(directory / "state_status.json")
        expected = {"schema_version": "ep21.participant_training_state.v1",
                    "status": "complete_participant_training_state_only", "participant": meta["participant"],
                    "training_role": "subject_unique", "training_images": 4712,
                    "training_repetitions": meta["rows"], "native_voxels": meta["native_count"],
                    "native_index_source_sha256": meta["mask_index_sha256"],
                    "shared_state": "shared_training_state.npz", "training_row_binding": "training_row_binding.json",
                    "regular_feature_values_read": False, "tau_train_values_read": False,
                    "tau_test_values_read": False, "OOD_access": False,
                    "predictions_or_scores_computed": False, "complete_decoder": False}
        require(all(record.get(key) == value for key, value in expected.items()), "completed_own_participant_state_required")
        eligible_count = record.get("eligible_voxels")
        require(type(eligible_count) is int and 500 <= eligible_count <= meta["native_count"],
                "actual_eligible_voxel_count_required")
        rows = prefix.read_json(directory / "training_row_binding.json")
        require(rows == {"training_image_ids": meta["ids"], "preserved_trial_rows_per_image": meta["groups"],
                         "role": "subject_unique", "participant": meta["participant"], "image_mean_weighting": "equal_images"},
                "exact_own_participant_image_and_repetition_binding_required")
    return _binding(meta, eligible_count)


def _array_headers(binding):
    np = prefix.np
    n, eligible, native = len(binding.image_ids), binding.eligible_count, binding.native_count
    shapes = {"normalized_X": (n, eligible), "penalized_Gram": (eligible + 1, eligible + 1),
              "all_native_voxel_ids": (native,), "eligible_native_voxel_ids": (eligible,),
              "eligible_mask": (native,), "neural_mean": (1, native), "neural_std": (1, native),
              "image_repetition_counts": (n,), "training_image_ids": (n,)}
    require(binding.state_path.is_file() and not binding.state_path.is_symlink()
            and binding.state_path.stat().st_size < 1_100_000_000, "bounded_retained_state_file_required")
    with zipfile.ZipFile(binding.state_path) as archive:
        require(set(archive.namelist()) == {key + ".npy" for key in ARRAY_NAMES}
                and len(archive.namelist()) == len(ARRAY_NAMES), "exact_state_arrays_required")
        for key, shape in shapes.items():
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "supported_state_array_header_required")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                actual, _, dtype = reader(stream)
                require(actual == shape and not dtype.hasobject, "state_array_shape_or_dtype_mismatch")
                if key in ("normalized_X", "penalized_Gram", "neural_mean", "neural_std"):
                    require(dtype.str == "<f4", "state_source_float32_required")
                elif key in ("all_native_voxel_ids", "eligible_native_voxel_ids", "image_repetition_counts"):
                    require(dtype.kind in "iu" and dtype.itemsize <= 8, "state_integer_array_required")
                elif key == "eligible_mask":
                    require(dtype.kind == "b", "state_boolean_eligibility_required")
                else:
                    require(dtype == np.asarray(binding.image_ids).dtype, "exact_image_identity_string_dtype_required")


def load_shared(participant, authorization_path, budget):
    """Load only an already completed own-participant state; never build/replay it."""
    authorize_full(authorization_path)
    require_compute(budget)
    meta = admit_participant(participant)
    binding = _admit_state_binding(meta)
    _runtime(meta["lock"])
    np = prefix.np
    _array_headers(binding)
    with np.load(binding.state_path, allow_pickle=False) as state:
        require(state["training_image_ids"].tolist() == list(binding.image_ids)
                and state["image_repetition_counts"].tolist() == list(binding.repetition_counts),
                "state_image_repetition_identity_mismatch")
        native, eligible, mask = (state[key] for key in
                                  ("all_native_voxel_ids", "eligible_native_voxel_ids", "eligible_mask"))
        require(bool((native[1:] > native[:-1]).all()) and int(mask.sum()) == binding.eligible_count
                and np.array_equal(native[mask], eligible), "own_native_ID_eligibility_binding_required")
        means, std = state["neural_mean"], state["neural_std"]
        require(bool(np.isfinite(means).all()) and bool(np.isfinite(std).all())
                and bool((std >= 0).all()) and np.array_equal(std[0] > 0, mask),
                "retained_training_moments_and_eligibility_required")
        x, gram = state["normalized_X"], state["penalized_Gram"]
    # No Gram recomputation and no order/layout conversion: preserve the arrays
    # saved by the source-math preparation. Bound finite checks to small rows.
    for array in (x, gram):
        for first in range(0, len(array), 64):
            budget.check()
            require(bool(np.isfinite(array[first:first + 64]).all()), "nonfinite_retained_training_state")
    augmented = np.hstack((x, np.ones((len(x), 1), dtype=np.float32)))
    return x, augmented, gram, eligible, binding
