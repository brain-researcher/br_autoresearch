"""Inactive recovery composition for a closed failed amended decode.

No CLI, scheduler action or access grant is provided. The active decode and all
its imports remain unchanged. An independently admitted caller must establish
that job47586225 failed, its HDF writer closed successfully, and its frozen inputs
and 40 producer outputs are unchanged. The old HDF remains read-only; selected
whole owner/layer groups are copied to a fresh original-format collection.

The original writer's flags record completion intent before its final flush;
reuse also requires ordinary closed-file, header and full payload validation.
They are not a transaction journal or model digests. Model provenance remains
the original procedural canonical-path and producer-job binding. A readable
file, allocated datasets or nonzero bytes do
not establish completion. Inconsistent committed data fail closed; no native
HDF repair, flag clearing, partial-block salvage or score-based reuse occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import h5py
import numpy as np

import amended_distributed_readout as active


OPERATION = "run_amended_finding_a_decode_recovery"
SOURCE_JOB = "47586225"
SOURCE_ROOT = active.models.BASE / "outputs/runs/finding_a_feature_decoding/amended_full001"
SOURCE_STAGE = SOURCE_ROOT / "decoded_point.h5"
SOURCE_PARTIAL = SOURCE_ROOT / "decoded_point.h5.partial"
SOURCE_CLAIM = SOURCE_ROOT / "decode/claim.json"
MODEL_POLICY = active.MODEL_POLICY
original, require = active.collection, active.require
TERMINAL_FAILURES = frozenset(("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
    "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "DEADLINE"))
_ROWS = 256


@dataclass(frozen=True)
class RecoveryResult:
    artifacts: tuple[Path, ...]
    recovery_summary: dict


def validate_source_claim(claim, closure):
    """Check coordinator-supplied operational evidence before any HDF open.

    This record does not inspect Slurm or independently prove writer closure.
    The admitting coordinator must obtain terminal evidence and check frozen
    sources; a fabricated or speculative closure record is not authorization.
    """
    require(isinstance(claim, dict) and isinstance(closure, dict),
            "original_claim_and_coordinator_closure_required")
    require(claim.get("phase") == "decode" and claim.get("batch_index") is None
        and claim.get("operation") == active.DECODE
        and claim.get("model_policy") == MODEL_POLICY
        and claim.get("allocation", {}).get("job_id") == SOURCE_JOB,
        "exact_original_amended_decode_claim_required")
    binding = claim.get("binding", {}).get("decode", {})
    active_jobs, completed_jobs = binding.get("afterok_job_ids"), binding.get("completed_model_job_ids")
    require(binding.get("job_id") == SOURCE_JOB and isinstance(active_jobs, list)
        and isinstance(completed_jobs, list), "original_40_model_dependency_binding_required")
    jobs = active_jobs + completed_jobs
    require(len(jobs) == len(set(jobs)) == 40
        and all(isinstance(job, str) and job.isdigit() and job != SOURCE_JOB for job in jobs),
        "40_distinct_original_producer_jobs_required")
    require(closure.get("source_job_id") == SOURCE_JOB
        and closure.get("terminal_state") in TERMINAL_FAILURES
        and closure.get("writer_closed") is True
        and closure.get("hdf_close_completed") is True
        and closure.get("frozen_sources_unchanged") is True
        and closure.get("native_file_failure") is False
        and closure.get("hdf_corruption_reported") is False
        and closure.get("original_model_job_ids") == sorted(jobs)
        and all(isinstance(closure.get(key), str) and closure[key].strip()
                for key in ("recorded_at_utc", "authority", "terminal_evidence",
                            "writer_closure_evidence", "hdf_close_evidence", "frozen_source_evidence")),
        "actual_failed_closed_job_unchanged_sources_and_no_native_file_failure_required")
    return tuple(sorted(jobs))


def _dataset(handle, path, shape, dtype):
    require(path in handle, "required_recovery_dataset_missing: " + path)
    item = handle[path]
    require(isinstance(item, h5py.Dataset) and item.shape == shape
        and item.dtype == np.dtype(dtype), "recovery_dataset_shape_or_dtype_changed: " + path)
    return item


def _true_attribute(group, name):
    value = group.attrs.get(name)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def validate_header(source, writer):
    """Bind the old partial to newly admitted original identities and axes."""
    writer._check()
    require(source.attrs.get("role") == "tau_test"
        and source.attrs.get("model_policy") == MODEL_POLICY
        and not bool(source.attrs.get("complete", True))
        and not bool(source.attrs.get("decoded_point_complete", False))
        and "staging" not in source and "resampling" not in source,
        "only_unpublished_original_amended_partial_can_be_recovered")
    require(source.attrs.get("feature_identity") == writer._feature_identity,
            "original_feature_identity_changed")
    require("image_ids" in source and source["image_ids"].shape == (224,)
        and tuple(source["image_ids"].asstr()[...].tolist()) == writer.image_ids,
        "original_tau_test_image_order_changed")
    for owner in original.PARTICIPANTS:
        writer._check()
        location = "decoders/" + owner
        require(location in source, "original_decoder_owner_missing")
        group = source[location]
        require("training_image_ids" in group and group["training_image_ids"].shape == (4712,)
            and tuple(group["training_image_ids"].asstr()[...].tolist()) == writer.training[owner],
            "original_decoder_training_image_order_changed")
        for name in original.prediction._AXIS_FIELDS:
            expected = getattr(writer.axes[owner], name)
            item = _dataset(source, location + "/" + name, expected.shape, expected.dtype)
            require(np.array_equal(item[...], expected), "original_decoder_native_axis_or_moments_changed")


def _owner_slots(writer, owner, layer):
    require(owner in original.PARTICIPANTS and layer in original.LAYERS,
            "original_owner_and_layer_required")
    streams = tuple(stream for stream in original.STREAMS
                    if original.metrics._owner(*stream) == owner)
    require(len(streams) == 9 and all(writer.fitted_object_ids[stream] is not None for stream in streams),
            "all_nine_original_owner_streams_required")
    return tuple(writer._slot(*stream, layer) for stream in streams)


def _coordinates(layer, first, last, c_ids, f_ids, seen):
    units = original.metrics._unit_count(layer)
    require(c_ids.shape == f_ids.shape == (last - first,)
        and np.all((c_ids >= 0) & (c_ids < units))
        and np.all((f_ids >= 0) & (f_ids < units))
        and np.unique(c_ids).size == last - first and not seen[c_ids].any(),
        "committed_group_has_missing_or_duplicate_coordinates")
    shape = original.assembly.coverage.prefix.SHAPES[original.LAYERS.index(layer)]
    if len(shape) == 1:
        expected_c = f_ids
    else:
        channel, spatial = f_ids % shape[0], f_ids // shape[0]
        expected_c = channel * (shape[1] * shape[2]) + (spatial % shape[1]) * shape[2] + spatial // shape[1]
    require(np.array_equal(c_ids, expected_c), "committed_group_F_C_coordinates_disagree")
    seen[c_ids] = True


def copy_committed_owner_layer(source, writer, *, owner, layer):
    """Return False for an uncommitted group; validate/copy a complete group.

    The nine stream flags and shared moment flag are the original writer's
    completion intent; successful HDF close is a separate entry prerequisite.
    A marked-complete but invalid group raises; it is not
    silently relabeled as incomplete. New groups stay incomplete until all
    bounded copies and validation finish. Original point-score groups, if any,
    are ignored and never copied.
    """
    writer._check()
    slots = _owner_slots(writer, owner, layer)
    moment_path = "decoders/" + owner + "/feature_moments/" + layer
    if moment_path not in source or not _true_attribute(source[moment_path], "complete"):
        return False
    if any(location not in source or not _true_attribute(source[location], "complete")
           for _, location in slots):
        return False
    with writer._operation():
        for key, _ in slots:
            writer._new_slot(key)
        units = original.metrics._unit_count(layer)
        coordinate_path = "feature_axes/" + layer + "/full_layer_C_index"
        coordinate = _dataset(source, coordinate_path, (units,), np.int64)
        source_moments = {name: _dataset(source, moment_path + "/" + name, (units,), np.float32)
                          for name in ("mean", "std")}
        existing = []
        for _, location in slots:
            existing.append({name: _dataset(source, location + "/" + name, shape, dtype)
                for name, shape, dtype in (
                    ("source_file_features", (units, 224), np.float32),
                    ("valid_mask", (units, 224), bool),
                    ("full_layer_C_index", (units,), np.int64),
                    ("full_layer_Fortran_index", (units,), np.int64))})
        if coordinate_path not in writer._h5:
            axis = writer._h5.create_dataset(coordinate_path, (units,), dtype="<i8")
        else:
            axis = writer._h5[coordinate_path]
        moments = writer._h5.require_group(moment_path)
        moments.attrs["complete"] = False
        for name in source_moments:
            moments.create_dataset(name, (units,), dtype="<f4")
        targets = []
        for _, location in slots:
            group = writer._h5.create_group(location)
            group.attrs["complete"] = False
            group.create_dataset("source_file_features", (units, 224), dtype="<f4", chunks=(min(256, units), 224))
            group.create_dataset("valid_mask", (units, 224), dtype="?", chunks=(min(256, units), 224))
            for name in ("full_layer_C_index", "full_layer_Fortran_index"):
                group.create_dataset(name, (units,), dtype="<i8")
            targets.append(group)
        seen = np.zeros(units, dtype=bool)
        for first in range(0, units, _ROWS):
            writer._check()
            last = min(first + _ROWS, units)
            global_c = coordinate[first:last]
            require(np.array_equal(global_c, np.arange(first, last)),
                    "committed_full_layer_C_axis_changed")
            axis[first:last] = global_c
            for name, item in source_moments.items():
                values = item[first:last]
                require(np.isfinite(values).all() and (name != "std" or np.all(values > 0)),
                        "committed_feature_moments_invalid")
                moments[name][first:last] = values
            c_ids = existing[0]["full_layer_C_index"][first:last]
            f_ids = existing[0]["full_layer_Fortran_index"][first:last]
            _coordinates(layer, first, last, c_ids, f_ids, seen)
            for saved, target in zip(existing, targets):
                writer._check()
                require(np.array_equal(saved["full_layer_C_index"][first:last], c_ids)
                    and np.array_equal(saved["full_layer_Fortran_index"][first:last], f_ids),
                    "shared_owner_stream_append_coordinates_changed")
                values, valid = saved["source_file_features"][first:last], saved["valid_mask"][first:last]
                require(np.isfinite(values).all() and valid.all(),
                        "committed_predictions_or_complete_valid_mask_invalid")
                target["source_file_features"][first:last] = values
                target["valid_mask"][first:last] = valid
                target["full_layer_C_index"][first:last] = c_ids
                target["full_layer_Fortran_index"][first:last] = f_ids
        require(seen.all(), "complete_owner_layer_coordinate_coverage_required")
        moments.attrs["complete"] = True
        for target in targets:
            target.attrs["complete"] = True
        writer._h5.flush()
        writer._check()
        for key, location in slots:
            record = writer._record(key, location, "decoded")
            record.update(stored_shape=[units, 224], storage_axes=["append_order_unit", "image"],
                feature_moments=moment_path, coordinates={name: location + "/" + name for name in
                ("full_layer_C_index", "full_layer_Fortran_index")},
                values=location + "/source_file_features", valid_mask=location + "/valid_mask")
    return True


def _stat_identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _claim():
    require(SOURCE_CLAIM.resolve() == SOURCE_CLAIM.absolute() and SOURCE_CLAIM.is_file(),
            "fixed_regular_original_claim_required")
    with SOURCE_CLAIM.open("rb") as stream:
        value = stream.read(65537)
    require(len(value) <= 65536, "bounded_original_operational_claim_required")
    return json.loads(value)


def recover_decoded_point(execution_context, *, require_admitted_context, budget,
        stage_h5, maximum_bytes, maximum_source_plus_destination_bytes,
        authenticated_tau_train_image_ids, closure_record):
    """New admitted entrypoint; old running caller cannot invoke this operation."""
    path = Path(stage_h5)
    require(path.is_absolute() and path.suffix == ".h5" and path.parent.is_dir()
        and path.resolve() == path.absolute() and path.parent != SOURCE_ROOT,
        "fresh_separate_recovery_stage_required")
    require(type(maximum_bytes) is int and maximum_bytes > 0
        and type(maximum_source_plus_destination_bytes) is int
        and maximum_source_plus_destination_bytes > maximum_bytes,
        "explicit_source_plus_fresh_destination_storage_allowance_required")
    options = dict(collection_path=path, maximum_bytes=maximum_bytes,
        maximum_source_plus_destination_bytes=maximum_source_plus_destination_bytes,
        source_partial=SOURCE_PARTIAL, source_claim=SOURCE_CLAIM,
        source_job_id=SOURCE_JOB, model_policy=MODEL_POLICY,
        closure_record=json.loads(json.dumps(closure_record, sort_keys=True, allow_nan=False)))
    check_admission = active.original._admission(execution_context, require_admitted_context,
        budget, OPERATION, options)
    model_jobs = validate_source_claim(_claim(), options["closure_record"])
    require(budget.allocation["job_id"] != SOURCE_JOB, "recovery_requires_a_new_allocation")
    require(SOURCE_PARTIAL.resolve() == SOURCE_PARTIAL.absolute() and SOURCE_PARTIAL.is_file(),
            "fixed_original_regular_partial_required")
    for reserved in (SOURCE_STAGE, SOURCE_STAGE.with_suffix(".decoded_point.json"),
            SOURCE_STAGE.with_suffix(".finalization_started.json")):
        require(not os.path.lexists(reserved), "published_or_finalizing_stage_requires_different_recovery")
    original._require_file_limit(maximum_bytes)
    source_identity = _stat_identity(SOURCE_PARTIAL)
    require(source_identity[2] + maximum_bytes <= maximum_source_plus_destination_bytes,
            "old_partial_and_new_complete_collection_exceed_selected_storage")
    bound = None

    def check():
        check_admission()
        if bound is not None:
            bound.check()

    check()
    manifest = active.assembly.load_complete_manifest(active.assembly.BASE / active.assembly.MANIFEST, budget=budget)
    axes, training = active._decoder_metadata(manifest, budget, check)
    reused, recomputed = [], []
    try:
        bound = active._bind_inputs(execution_context, require_admitted_context=require_admitted_context, budget=budget)
        prepared = active._prepare_models(bound, axes, training,
            authenticated_tau_train_image_ids=authenticated_tau_train_image_ids,
            execution_context=execution_context, require_admitted_context=require_admitted_context, budget=budget)
        prepared.check()
        require(prepared.model_policy == MODEL_POLICY and not prepared.unavailable_reasons
            and set(prepared.prepared) == set(original.STREAMS)
            and dict(prepared.fitted_object_ids) == active._expected_fitted(),
            "same_complete_45_stream_amended_model_panel_required")
        # Native open failures propagate: never clear an HDF consistency flag.
        with h5py.File(SOURCE_PARTIAL, "r", driver="sec2") as source:
            with active.publication.AmendedStagedCollection(path, image_ids=bound.image_ids,
                    decoder_axes_by_owner=axes, decoder_training_image_ids_by_owner=training,
                    fitted_object_ids=prepared.fitted_object_ids, feature_identity=bound.feature_identity,
                    budget=budget, maximum_bytes=maximum_bytes, require_admitted=check) as writer:
                validate_header(source, writer)
                for layer in original.LAYERS:
                    for owner in original.PARTICIPANTS:
                        check()
                        require(_stat_identity(SOURCE_PARTIAL) == source_identity,
                                "closed_source_partial_changed_during_recovery")
                        prepared.check()
                        if copy_committed_owner_layer(source, writer, owner=owner, layer=layer):
                            reused.append((owner, layer))
                        else:
                            members = {stream: value for stream, value in prepared.prepared.items()
                                       if value.decoder_axis.participant == owner}
                            writer.write_owner_layer(owner=owner, layer=layer, prepared_by_stream=members,
                                blocks=active._decoder_blocks(manifest, axes[owner], layer, budget, check))
                            recomputed.append((owner, layer))
                writer.seal_decoded()
                require(len(reused) + len(recomputed) == 95 and len(writer._records) == 855,
                        "all_95_owner_layers_and_855_slots_required")
                prepared = None
                def provider(**kwargs):
                    check()
                    return writer.block_provider(**kwargs, truth_provider=bound.truth_reader)
                point = active.inference.evaluate_panel(provider, training, bound.image_ids,
                    check=check, result_sink=writer.result_sink)
                check()
                require(_stat_identity(SOURCE_PARTIAL) == source_identity,
                        "original_partial_must_remain_unchanged")
                artifacts = writer.publish_decoded_point(point)
        return RecoveryResult(tuple(map(Path, artifacts)), {
            "source_job_id": SOURCE_JOB, "source_partial": str(SOURCE_PARTIAL),
            "original_model_job_ids": model_jobs, "model_policy": MODEL_POLICY,
            "reused_owner_layers": reused, "recomputed_owner_layers": recomputed,
            "decoded_slots": 855, "unavailable_slots": 0, "all_point_scores_recomputed": True,
            "source_partial_preserved": True,
            "provenance": "original canonical model paths and producer jobs with coordinator-verified frozen sources; no per-group model digest was stored",
            "quantitative_finding": "not_inspected_in_operational_report"})
    finally:
        if bound is not None:
            bound.close()
