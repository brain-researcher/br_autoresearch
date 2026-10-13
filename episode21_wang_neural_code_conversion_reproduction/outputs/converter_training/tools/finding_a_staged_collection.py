"""Dormant decoded-point publication, read-only replay and one-shot completion.

The successful stage retains one HDF5 inode and a small immutable JSON record.
It has all855 terminal slots and every decoded point vector, but no intervals
or final scientific report. Read workers reconstruct saved metadata only; no
model, neural input or original decoder coefficient is reopened here.

After the caller establishes that all readers are closed, finalization claims
a permanent marker and creates a hardlink at a NEW final.partial path. The
unchanged collection publisher then adds resampling arrays and final sidecars
on this same inode. The original stage path/history remain present. Marker
existence permanently disables stage replay, including after failed completion;
there is no resume, retry or overwrite. Historical stage JSON describes its
publication phase, not the current inference status after finalization.

Admission, reader coordination and fatal RLIMIT_FSIZE/SIGXFSZ policy are caller
prerequisites. The caller's aggregate storage check must count each encountered
(st_dev,st_ino) once while stage/final aliases coexist, plus all sidecar bytes.
No resource grant, process isolation or scheduler is supplied by this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType

import h5py
import numpy as np

import finding_a_collection as original
from finding_a_grouped_collection import GroupedFindingACollection
import finding_a_draw_artifacts as codec


METADATA_BYTES = 16 * 1024**2
_METADATA_DATASET = "staging/decoded_point_json"
_KIND = "finding_a_decoded_point_stage"


def _metadata_path(path):
    return path.with_suffix(".decoded_point.json")


def _marker_path(path):
    return path.with_suffix(".finalization_started.json")


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encoded(value):
    return json.dumps(original._json(value), sort_keys=True, allow_nan=False).encode("utf-8")


def _view(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _view(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_view(item) for item in value)
    return value


def _identity(state):
    return codec.stage_identity_record(dict(collection_path=str(state.path),
        point_path=str(_metadata_path(state.path)) + "#point",
        fitted_object_ids=[dict(method=method, source=source, target=target,
            fitted_object_id=state.fitted_object_ids[(method, source, target)])
            for method, source, target in original.STREAMS],
        feature_identity=json.loads(state._feature_identity)), check=state._check)


def _bind_point(state, point):
    # The codec validates the original point roster, identity draw and unchanged
    # compact reductions. These are the same saved-vector checks as the point
    # portion of FindingACollection._bind_results, without placeholder intervals.
    record = codec.panel_to_record(point, check=state._check)
    original._require(point.image_ids == state.image_ids, "stage_point_image_identity_mismatch")
    original._require(set(state._records) == set(original.SLOTS), "all_855_stage_slots_required")
    decoded = {key for key, value in state._records.items() if value["status"] == "decoded"}
    original._require(set(state._scores) == decoded, "all_decoded_stage_point_vectors_required")
    for key, saved_slot in state._records.items():
        state._check()
        method, source, target, layer = key
        position = (original.LAYERS.index(layer), original.PARTICIPANTS.index(source))
        if target is not None:
            position += (original.PARTICIPANTS.index(target),)
        for metric in ("pattern", "profile"):
            values = getattr(point, metric)
            actual = values.layer_means[method][position]
            if saved_slot["status"] == "unavailable":
                original._require(np.isnan(actual) and values.layer_reasons[method][position] == saved_slot["reason"],
                                  "unavailable_stage_slot_not_preserved_in_point")
            else:
                saved = state._scores[key][metric]
                original._require((np.isnan(actual) if saved["mean"] is None else actual == saved["mean"])
                    and values.layer_defined_counts[method][position] == saved["defined_count"]
                    and values.layer_undefined_counts[method][position] == saved["undefined_count"],
                    "stage_point_disagrees_with_saved_vectors")
    return record


class StagedFindingACollection(GroupedFindingACollection):
    def __init__(self, path, **kwargs):
        kwargs["require_admitted"]()
        path = Path(path)
        for reserved in (_metadata_path(path), _marker_path(path)):
            if os.path.lexists(reserved) or os.path.lexists(str(reserved) + ".partial"):
                raise FileExistsError(reserved)
        super().__init__(path, **kwargs)

    def publish(self, *args, **kwargs):
        raise ValueError("staged_collection_requires_publish_decoded_point")

    def publish_decoded_point(self, point):
        with self._operation():
            original._require(self._sealed, "sealed_complete_roster_required")
            point_record = _bind_point(self, point)
            record = dict(kind=_KIND, complete=False, decoded_point_complete=True,
                inference="pending", stage_identity=_identity(self), point=point_record,
                records=[self._records[key] for key in original.SLOTS],
                scores=[dict(slot=list(key), summary=self._scores[key])
                        for key in original.SLOTS if key in self._scores])
            encoded = _encoded(record)
            original._require(len(encoded) <= METADATA_BYTES, "bounded_stage_metadata_required")
            dataset = self._h5.create_dataset(_METADATA_DATASET, (len(encoded),), dtype="u1")
            for first in range(0, len(encoded), original._IO_BYTES):
                self._check()
                dataset[first:first + original._IO_BYTES] = np.frombuffer(
                    encoded[first:first + original._IO_BYTES], dtype=np.uint8)
            self._h5.attrs["decoded_point_complete"] = True
            self._h5.attrs["complete"] = False
            self._h5.flush()
            self._check()
            self.close()
            original._require_file_limit(self.maximum_bytes)
            used = self.partial.stat().st_size
            original._require(used + len(encoded) <= self.maximum_bytes,
                              "collection_exceeds_caller_aggregate_byte_cap")
            os.link(self.partial, self.path)
            self.partial.unlink()
            metadata = _metadata_path(self.path)
            self._sidecar(metadata, record, used)
            _sync_directory(self.path.parent)
            return self.path, metadata


class _RestoredCollection(GroupedFindingACollection):
    def _check(self):
        super()._check()
        if self._replay_only:
            original._require(not os.path.lexists(self._stage_marker), "stage_finalization_already_claimed")

    def _sidecar(self, path, record, used):
        if self._finalizing and path == self.manifest_path:
            # Stage and final HDF paths share the same inode, already counted by
            # original publish. Count the other retained files exactly once.
            used += self._stage_metadata.stat().st_size + self._stage_marker.stat().st_size
        return super()._sidecar(path, record, used)


class DecodedPointReader:
    """Read-only public surface; no writer or finalizer methods are exposed."""
    def __init__(self, state, point, identity):
        self._state = state
        self.point, self.stage_identity = point, _view(identity)
        self.image_ids, self.training = state.image_ids, state.training
        self.axes = MappingProxyType(dict(state.axes))
        self.feature_identity = _view(json.loads(state._feature_identity))
        self.fitted_object_ids = MappingProxyType(dict(state.fitted_object_ids))
        self.records = _view(state._records)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self._state._abort(value)
        else:
            self.close()

    def block_provider(self, **kwargs):
        return self._state.block_provider(**kwargs)

    def close(self):
        self._state.close()


def _dataset(state, path, shape, dtype):
    item = state._h5[path]
    original._require(isinstance(item, h5py.Dataset) and item.shape == shape
                      and item.dtype == np.dtype(dtype), "stage_dataset_header_mismatch")


def _restore_slots(state, record):
    slots = record["records"]
    original._require(isinstance(slots, list) and len(slots) == len(original.SLOTS),
                      "all_855_stage_records_required")
    for key, saved in zip(original.SLOTS, slots):
        state._check()
        _, location = state._slot(*key)
        status, reason = saved["status"], saved["reason"]
        original._require(status in ("decoded", "unavailable"), "terminal_stage_slot_required")
        original._require((status == "decoded" and reason is None) or
            (status == "unavailable" and isinstance(reason, str) and 0 < len(reason) <= 1024),
            "explicit_stage_slot_status_required")
        expected = state._record(key, location, status, reason)
        if status == "decoded":
            units = original.metrics._unit_count(key[3])
            owner = original.metrics._owner(*key[:3])
            moment = "decoders/" + owner + "/feature_moments/" + key[3]
            expected.update(stored_shape=[units, 224], storage_axes=["append_order_unit", "image"],
                feature_moments=moment, coordinates={name: location + "/" + name for name in
                ("full_layer_C_index", "full_layer_Fortran_index")},
                values=location + "/source_file_features", valid_mask=location + "/valid_mask")
            original._require(state.fitted_object_ids[key[:3]] is not None
                and bool(state._h5[location].attrs["complete"])
                and bool(state._h5[moment].attrs["complete"]), "complete_stage_values_and_moments_required")
            _dataset(state, expected["values"], (units, 224), np.float32)
            _dataset(state, expected["valid_mask"], (units, 224), bool)
            for path in expected["coordinates"].values():
                _dataset(state, path, (units,), np.int64)
            _dataset(state, "feature_axes/" + key[3] + "/full_layer_C_index", (units,), np.int64)
            for name in ("mean", "std"):
                _dataset(state, moment + "/" + name, (units,), np.float32)
        original._require(original._json(expected) == saved, "stage_slot_metadata_mismatch")
    scores = record["scores"]
    original._require(isinstance(scores, list), "stage_point_scores_required")
    for entry in scores:
        state._check()
        key = tuple(entry["slot"])
        original._require(key in state._records and state._records[key]["status"] == "decoded"
                          and key not in state._scores, "unique_decoded_point_slot_required")
        summary = entry["summary"]
        original._require(set(summary) == {"pattern", "profile"}, "both_point_metric_vectors_required")
        location = state._records[key]["hdf5_group"]
        units = original.metrics._unit_count(key[3])
        for metric, count in (("pattern", 224), ("profile", units)):
            value = summary[metric]
            original._require(set(value) == {"mean", "defined_count", "undefined_count",
                "correlations", "defined_mask", "coordinate"}
                and value["correlations"] == "/" + location + "/" + metric + "/correlations"
                and value["defined_mask"] == "/" + location + "/" + metric + "/defined_mask"
                and value["coordinate"] == ("image_ids" if metric == "pattern" else
                    "feature_axes/" + key[3] + "/full_layer_C_index"), "stage_point_vector_paths_mismatch")
            _dataset(state, value["correlations"], (count,), np.float64)
            _dataset(state, value["defined_mask"], (count,), bool)
        state._scores[key] = summary


def open_decoded_point(stage_h5, *, budget, maximum_bytes, require_admitted):
    """Reopen only an admitted successful stage; never a partial or claimed one."""
    original._require(callable(require_admitted), "external_stage_admission_required")
    require_admitted()
    original.assembly._require_reader_context(budget)
    original._require(type(maximum_bytes) is int and maximum_bytes > 0, "explicit_stage_byte_cap_required")
    original._require_file_limit(maximum_bytes)
    path = Path(stage_h5)
    metadata, marker = _metadata_path(path), _marker_path(path)
    original._require(path.is_absolute() and path.suffix == ".h5" and not path.is_symlink()
        and not metadata.is_symlink() and path.is_file() and metadata.is_file(), "published_stage_hdf_and_metadata_required")
    original._require(not os.path.lexists(marker), "stage_finalization_already_claimed")
    original._require(metadata.stat().st_size <= METADATA_BYTES
        and path.stat().st_size + metadata.stat().st_size <= maximum_bytes, "bounded_stage_files_required")
    with metadata.open("rb") as stream:
        encoded = stream.read(METADATA_BYTES + 1)
    original._require(len(encoded) <= METADATA_BYTES, "bounded_stage_metadata_required")
    record = json.loads(encoded)
    original._require(record["kind"] == _KIND and record["complete"] is False
        and record["decoded_point_complete"] is True and record["inference"] == "pending",
        "successful_decoded_point_stage_required")
    identity = codec.stage_identity_record(record["stage_identity"], check=budget.check)
    original._require(identity["collection_path"] == str(path)
        and identity["point_path"] == str(metadata) + "#point", "exact_stage_path_identity_required")
    state = object.__new__(_RestoredCollection)
    state.path, state.partial = path, path.with_name(path.name + ".partial")
    state.manifest_path = path.with_suffix(".decoded_feature_manifest.json")
    state.report_path = path.with_suffix(".finding_a_report.json")
    state._stage_metadata, state._stage_marker = metadata, marker
    state._replay_only, state._finalizing = True, False
    state.budget, state.maximum_bytes, state.require_admitted = budget, maximum_bytes, require_admitted
    state._closing = state._failed = state._closed = False
    state._sealed = True
    state._h5 = None
    state._records, state._scores = {}, {}
    try:
        state._check()
        state._h5 = h5py.File(path, "r", driver="sec2")
        original._require(state._h5.attrs["role"] == "tau_test"
            and not bool(state._h5.attrs["complete"])
            and bool(state._h5.attrs["decoded_point_complete"])
            and "resampling" not in state._h5, "unfinalized_complete_decoded_point_hdf_required")
        stored = state._h5[_METADATA_DATASET]
        original._require(stored.ndim == 1 and stored.dtype == np.dtype("uint8")
                          and 0 < stored.size <= METADATA_BYTES, "bounded_embedded_stage_metadata_required")
        original._require(json.loads(stored[...].tobytes()) == record, "stage_hdf_json_binding_mismatch")
        state._feature_identity = original._identity(identity["feature_identity"])
        original._require(state._h5.attrs["feature_identity"] == state._feature_identity,
                          "stage_feature_identity_mismatch")
        original._require(state._h5["image_ids"].shape == (224,), "complete_stage_image_ID_axis_required")
        state.image_ids = tuple(state._h5["image_ids"].asstr()[...].tolist())
        training, axes = {}, {}
        for owner in original.PARTICIPANTS:
            state._check()
            group = state._h5["decoders/" + owner]
            original._require(group["training_image_ids"].shape == (4712,),
                              "complete_stage_training_ID_axis_required")
            training[owner] = tuple(group["training_image_ids"].asstr()[...].tolist())
            for name in original.prediction._AXIS_FIELDS:
                item = group[name]
                dtype = np.float32 if name in ("mean", "std") else np.int64
                original._require(item.ndim == 1 and item.dtype == np.dtype(dtype)
                    and 0 < item.size * item.dtype.itemsize <= METADATA_BYTES,
                    "bounded_stage_native_axis_metadata_required")
            axes[owner] = original.prediction._axis(original.prediction.NativeAxis(owner,
                *(group[name][...] for name in original.prediction._AXIS_FIELDS)), state._check)
        state.training, _ = original.inference._context(training, state.image_ids, None, state._check)
        state.axes = axes
        state.fitted_object_ids = {(row["method"], row["source"], row["target"]): row["fitted_object_id"]
                                  for row in identity["fitted_object_ids"]}
        _restore_slots(state, record)
        point = codec.panel_from_record(record["point"], expected_image_ids=state.image_ids, check=state._check)
        _bind_point(state, point)
        state._check()
        return DecodedPointReader(state, point, identity)
    except BaseException as error:
        state._abort(error)
        raise


def finalize_decoded_point(stage_h5, final_h5, primary, incoming, dyadic, *,
                           budget, maximum_bytes, require_admitted, require_readers_closed):
    """One sole-writer completion after external closure of all draw readers.

    ``require_readers_closed`` is a coordinator prerequisite, not a lock or a
    self-issued grant. The permanent marker prevents subsequent normal replay
    or retry. Stage, partial and final aliases must count as one storage inode.
    """
    original._require(callable(require_readers_closed), "external_reader_closure_required")
    with open_decoded_point(stage_h5, budget=budget, maximum_bytes=maximum_bytes,
                           require_admitted=require_admitted) as reader:
        state = reader._state
        final = Path(final_h5)
        original._require(final.is_absolute() and final.suffix == ".h5"
            and final != state.path and final.parent.is_dir(), "new_absolute_final_destination_required")
        destinations = (final, final.with_name(final.name + ".partial"),
            final.with_suffix(".decoded_feature_manifest.json"), final.with_suffix(".finding_a_report.json"))
        for path in destinations:
            if os.path.lexists(path) or os.path.lexists(str(path) + ".partial"):
                raise FileExistsError(path)
        original._require(codec.panel_to_record(primary.point, check=state._check)
                          == codec.panel_to_record(reader.point, check=state._check),
                          "final_primary_point_must_equal_published_stage")
        state._bind_results(primary, incoming, dyadic)
        require_readers_closed()
        state._check()
        marker_record = _encoded(dict(stage_hdf=str(state.path), final_hdf=str(final),
                                      status="finalization_started_no_retry"))
        original._require(state.path.stat().st_size + state._stage_metadata.stat().st_size
            + len(marker_record) <= maximum_bytes, "collection_exceeds_caller_aggregate_byte_cap")
        with state._stage_marker.open("xb") as stream:
            stream.write(marker_record)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(state._stage_marker.parent)
        # Close the finalizer's own read handle before obtaining its write handle.
        reader.close()
        original._require_file_limit(maximum_bytes)
        state.require_admitted()
        stage_path = state.path
        state.path, state.partial = final, destinations[1]
        state.manifest_path, state.report_path = destinations[2:]
        state._replay_only, state._finalizing = False, True
        os.link(stage_path, state.partial)
        state._closed = False
        state._h5 = None
        try:
            state._check()
            state._h5 = h5py.File(state.partial, "r+", driver="sec2")
            original._require(not bool(state._h5.attrs["complete"])
                and bool(state._h5.attrs["decoded_point_complete"])
                and "resampling" not in state._h5, "unfinalized_stage_required_for_sole_writer")
            return original.FindingACollection.publish(state, primary, incoming, dyadic)
        except BaseException as error:
            state._abort(error)
            raise
