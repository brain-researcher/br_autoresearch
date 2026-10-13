"""Unactivated EP21 decoder assembly and bounded sparse coefficient consumption.

The full target is all 5 people x 19 layers = 74,306,440 units. A catalog is a
plan, never a completed decoder. Assembly scans every declared physical row and
its terminal record before publishing a complete manifest. Coefficients stay in
immutable fragments: no dense native-voxel x target matrix or duplicate weight
archive is materialized. Original prefix/channel and new grouped formats are
supported; only explicitly committed rows may be reused.

Empirical assembly requires the exact full-stage authority and compute admission.
The sparse reader is a library interface over a validated complete manifest and
the caller's verified compute budget, not another job or approval flow. It does
not authorize the caller's later prediction/converter/scoring stage. This module
does not read responses, predict, score, fit converters, or access tau/OOD. Pure
metadata and in-memory validators serve focused tests inside allocations.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from itertools import groupby
import math
import os
from pathlib import Path, PurePosixPath
import socket
import time
import zipfile

import decoder_coverage as coverage

BASE = Path(__file__).resolve().parents[3]
ROOT = "outputs/decoder_training"
FULL_ROOT = ROOT + "/full_continuation_001"
MANIFEST = FULL_ROOT + "/assembled_decoder_manifest.json"
AXES = "unit_by_ranked_selected_native_voxel; bias_separate"
NORMALIZATION = "role_assigned_float32_image_means_then_training_mean_sample_std_before_float32_fit"
MAX_BLOCK_UNITS = 256
MAX_NORMALIZATION_BYTES = 16 * 1024**2
FORMATS = ("retained_prefix", "continuation_fragment", "grouped_channels")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(path):
    require(isinstance(path, str), "relative episode-local path required")
    value = PurePosixPath(path)
    require(not value.is_absolute() and str(value) == path and ".." not in value.parts
            and value.parts[:2] == ("outputs", "decoder_training"),
            "only explicitly named decoder-training artifacts are permitted")
    return path


def physical_path(base, path):
    relative = safe_relative(path)
    result = Path(base) / relative
    require(result.resolve() == result.absolute(), "symlinked decoder artifacts are not admitted")
    return result


def span_from_record(record):
    start, stop = record["spatial_Fortran_interval"]
    return coverage.Span(record["participant"], record["layer"], record["channel"], start, stop).validate()


@dataclass(frozen=True)
class FragmentRef:
    path: str
    format: str
    group: str
    status_path: str
    span: coverage.Span

    def validate(self):
        self.span.validate()
        safe_relative(self.path)
        safe_relative(self.status_path)
        require(self.path.endswith(".h5") and self.status_path.endswith(".json"), "fragment/status file types required")
        require(self.format in FORMATS, "unknown decoder fragment format")
        require(self.group == (f"{self.span.layer}/channel_{self.span.channel:03d}"
                if self.format == "grouped_channels" else ""), "exact group binding required")
        if self.format == "retained_prefix":
            require(self.span.participant == "sub-01" and self.span.channel == 0
                    and 0 <= self.span.start < self.span.stop <= 64, "fixed retained prefix scope required")
        if self.format == "continuation_fragment":
            require(self.span.participant == "sub-01" and self.span.layer == "conv1_1"
                    and self.span.channel == 0 and 64 <= self.span.start < self.span.stop <= 50176,
                    "fixed channel continuation scope required")
        return self

    def record(self):
        self.validate()
        return {"path": self.path, "format": self.format, "group": self.group,
                "status_path": self.status_path, **self.span.record()}

    @classmethod
    def from_record(cls, record):
        return cls(record["path"], record["format"], record["group"],
                   record["status_path"], span_from_record(record)).validate()


@dataclass(frozen=True)
class StateRef:
    participant: str
    path: str
    binding_path: str
    status_path: str

    def record(self):
        require(self.participant in coverage.PARTICIPANTS, "unknown normalization participant")
        for value in (self.path, self.binding_path, self.status_path):
            safe_relative(value)
        require(self.path.endswith("shared_training_state.npz")
                and self.binding_path.endswith("training_row_binding.json"), "exact state/binding filenames required")
        return dict(participant=self.participant, path=self.path,
                    binding_path=self.binding_path, status_path=self.status_path)


def required_state(participant):
    directory = (ROOT + "/retained_prefix_001" if participant == "sub-01"
                 else FULL_ROOT + "/shared/" + participant)
    return StateRef(participant, directory + "/shared_training_state.npz",
                    directory + "/training_row_binding.json",
                    directory + ("/prefix_status.json" if participant == "sub-01" else "/state_status.json"))


def verify_catalog(fragments, states):
    """Exact structural scope only. Success does not prove any artifact exists."""
    fragments, states = tuple(fragments), tuple(states)
    for ref in fragments:
        ref.validate()
    count = coverage.verify_complete_coverage([ref.span for ref in fragments])
    require(count == coverage.TOTAL, "full five-participant nineteen-layer target required")
    require(len(states) == 5 and {state.participant for state in states} == set(coverage.PARTICIPANTS),
            "exactly one training normalization state per required participant")
    for state in states:
        require(state == required_state(state.participant), "fixed participant normalization binding required")
        state.record()
    return {"units": count, "participants": list(coverage.PARTICIPANTS),
            "layers": list(coverage.prefix.LAYERS), "physical_rows_validated": False,
            "complete_decoder": False}


def _catalog_remainder_spans(original, remainder, *, io_recovery=False):
    """Bind admitted physical pieces to one unchanged logical assignment."""
    work = original["work_item_id"]
    generation = "002" if work == "work_0013" else "001"
    output_directory = FULL_ROOT + f"/remainder_shards_{generation}/{work}"
    if io_recovery:
        import reviewed_io_recovery as recovery
        require(work in recovery.WORK_IDS, "only the five reviewed I/O assignments may recover")
        output_directory = FULL_ROOT + f"/io_remainder_shards_001/{work}"
    require(isinstance(remainder, dict) and remainder.get("work_item_id") == work
            and remainder.get("participant") == original["participant"]
            and remainder.get("predecessor_status") == FULL_ROOT + f"/shards/{work}/work_item_status.json"
            and remainder.get("output_directory") == output_directory,
            "exact original and remainder physical identities required")
    requested = [span_from_record(row) for row in original["spans"]]
    require(isinstance(remainder.get("committed_spans"), list)
            and isinstance(remainder.get("spans"), list), "explicit remainder partition required")
    committed = [span_from_record(row) for row in remainder["committed_spans"]]
    fresh = [span_from_record(row) for row in remainder["spans"]]
    whole = {span.key: span for span in requested}
    done, remaining = ({span.key: span for span in rows} for rows in (committed, fresh))
    require(len(whole) == len(requested) and len(done) == len(committed)
            and len(remaining) == len(fresh) and committed and fresh
            and set(done) <= set(whole) and set(remaining) <= set(whole)
            and list(done) == [key for key in whole if key in done]
            and list(remaining) == [key for key in whole if key in remaining],
            "remainder pieces must retain unique ordered original channel identities")
    for key, span in whole.items():
        saved = done.get(key)
        cursor = span.start if saved is None else saved.stop
        require((saved is None or saved.start == span.start) and span.start <= cursor <= span.stop
                and remaining.get(key) == (coverage.Span(*key, cursor, span.stop) if cursor < span.stop else None),
                "original committed and fresh remainder must exactly partition each assignment")
    require(type(remainder.get("units_requested")) is int
            and remainder["units_requested"] == sum(span.count for span in fresh)
            and sum(span.count for span in committed + fresh) == original["units_requested"],
            "original and remainder unit counts must agree")
    return committed, fresh


def planned_catalog(work_manifest, admitted_retained_spans, *, remainder_items=None, recovery_review=None):
    """Combine immutable retained work and explicitly planned generalized shards.

    The shared execution contract admits retained statuses and actual closed
    parent evidence before this function. Only exact committed spans from that
    admission enter the catalog, including an optional closed partial layer.
    An explicit remainder_items mapping comes from the separate remainder
    admission; this metadata-only catalog independently rechecks each logical
    partition. A separately admitted recovery_review splits only the five
    inspected I/O-failed assignments. Omission preserves original behavior.
    """
    if remainder_items is not None:
        import decoder_deadline_remainders as remainders
        require(isinstance(remainder_items, dict)
                and set(remainder_items) in (set(remainders.WORK_IDS), set(remainders.WORK_IDS_002))
                and set(remainder_items) <= {item["work_item_id"] for item in work_manifest["work_items"]},
                "only explicitly admitted original or cumulative remainder assignments are supported")
    recovered = {}
    if recovery_review is not None:
        import reviewed_io_recovery as recovery
        recovered = recovery.validate_review_structure(recovery_review, work_manifest)
        require(not set(recovered).intersection(remainder_items or {}),
                "deadline and I/O remainder assignments must be disjoint")
    fragments = [FragmentRef(ROOT + f"/retained_prefix_001/{layer}.h5", "retained_prefix", "",
                            ROOT + "/retained_prefix_001/prefix_status.json", span)
                 for layer, span in zip(coverage.prefix.LAYERS,
                        list(coverage.retained_spans())[:19])]
    fragments.append(FragmentRef(ROOT + "/continuation_001/conv1_1_channel0.h5",
        "continuation_fragment", "", ROOT + "/continuation_001/continuation_status.json",
        coverage.Span("sub-01", "conv1_1", 0, 64, 50176)))
    retained_entries = work_manifest["retained"]
    layer_entries = [entry for entry in retained_entries if entry["kind"] == "layer_continuation"]
    require(len(layer_entries) <= 1, "one explicitly closed retained layer attempt at most")
    extra_spans = [remainder for span in admitted_retained_spans
                   for remainder in coverage.subtract_retained(span)]
    require(not extra_spans or layer_entries, "extra retained rows require an admitted closed layer record")
    if layer_entries:
        status_path = safe_relative(layer_entries[0]["status"])
        require(status_path == ROOT + "/layer_continuation_001/continuation_status.json",
                "fixed optional layer record required")
        for span in extra_spans:
            fragments.append(FragmentRef(ROOT + "/layer_continuation_001/channels.h5",
                "grouped_channels", f"{span.layer}/channel_{span.channel:03d}", status_path, span))
    for item in work_manifest["work_items"]:
        item_id = item["work_item_id"]
        require(isinstance(item_id, str) and item_id and all(character.isalnum() or character in "_-" for character in item_id),
                "safe work-item identifier required")
        directory = FULL_ROOT + "/shards/" + item_id
        if remainder_items is not None and item_id in remainder_items:
            replacement = remainder_items[item_id]
            committed, fresh = _catalog_remainder_spans(item, replacement)
            parts = ((directory, committed), (replacement["output_directory"], fresh))
        elif item_id in recovered:
            replacement = recovered[item_id]
            committed, fresh = _catalog_remainder_spans(item, replacement, io_recovery=True)
            parts = ((directory, committed), (replacement["output_directory"], fresh))
        else:
            parts = ((directory, [span_from_record(row) for row in item["spans"]]),)
        for physical_directory, spans in parts:
            for span in spans:
                require(span.participant == item["participant"], "work-item participant mismatch")
                fragments.append(FragmentRef(physical_directory + "/channels.h5", "grouped_channels",
                    f"{span.layer}/channel_{span.channel:03d}", physical_directory + "/work_item_status.json", span))
    states = tuple(required_state(participant) for participant in coverage.PARTICIPANTS)
    verify_catalog(fragments, states)
    return tuple(fragments), states


def _admit_normalization(state, check):
    """Reuse the existing source-role metadata binding, without loading X/Gram."""
    import numpy as np
    import participant_training_state as state_builder
    metadata = state_builder.admit_participant(state.participant)
    admitted = state_builder._admit_state_binding(metadata)
    require(admitted.state_path == BASE / state.path
            and admitted.training_row_binding_path == BASE / state.binding_path,
            "normalization state path differs from admitted participant")
    result = load_normalization(BASE, state, check)
    require(len(result.native_ids) == admitted.native_count
            and len(result.eligible_ids) == admitted.eligible_count
            and result.training_image_ids.tolist() == list(admitted.image_ids)
            and np.array_equal(result.repetition_counts, admitted.repetition_counts),
            "loaded normalization differs from authenticated source-role binding")
    return result


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_terminal_record(ref, record, state, committed_index=None, *, recovery_review=None,
                             base_directory=None):
    """Bind selected rows to a closed worker record, including partial successes."""
    require(record.get("participant") == ref.span.participant
            and record.get("training_role") == "subject_unique"
            and record.get("training_images") == 4712, "terminal record training-role mismatch")
    for flag in ("predictions_or_scores_computed", "tau_train_values_read", "tau_test_values_read", "OOD_access"):
        require(record.get(flag) is False, "terminal record violates closed data/scoring boundaries")
    recovered_item = None
    if recovery_review is not None:
        import reviewed_io_recovery as recovery
        items = recovery.validate_review_structure(recovery_review)
        recovered_item = next((item for item in items.values()
                               if ref.status_path == item["predecessor_status"]), None)
    if recovered_item is None:
        require(not record.get("uncertain_publication") and not record.get("completed_units_is_lower_bound", False),
                "uncertain marker publication is not exact reusable coverage")
        require(not any(row.get("affects_fragment") for row in record.get("closeout_errors", [])),
                "fragment finalization must succeed before assembly")
    if ref.format == "retained_prefix":
        require(record.get("schema_version") == "ep21.retained_decoder_prefix.v1"
                and record.get("status") == "complete_retained_prefix_only"
                and record.get("completed_units") == 1216
                and record.get("shared_state") == "shared_training_state.npz", "completed prefix record required")
        specs = record["units"][ref.span.layer]
        expected = coverage.prefix.unit_specs(ref.span.layer, ref.span.shape)
        require(specs == [dict(value, status="complete") for value in expected], "prefix unit identities mismatch")
        expected_path = str(PurePosixPath(ref.status_path).parent / f"{ref.span.layer}.h5")
    elif ref.format == "continuation_fragment":
        require(record.get("schema_version") == "ep21.decoder_continuation.v1"
                and record.get("status") in ("complete_continuation_tranche_only", "partial_continuation_stopped"),
                "terminal channel record required")
        start, stop = record["committed_spatial_Fortran_interval"]
        require((record.get("layer"), record.get("channel")) == (ref.span.layer, ref.span.channel)
                and start <= ref.span.start < ref.span.stop <= stop, "channel rows not in exact committed coverage")
        expected_path = str(PurePosixPath(ref.status_path).parent / record["fragment"])
        require(record.get("shared_state_reference") == str(BASE / state.path), "channel normalization reference mismatch")
    else:
        require(record.get("schema_version") in ("ep21.decoder_layer_continuation.v1", "ep21.decoder_work_item.v1")
                and record.get("status") in ("complete_layer_continuation_only", "partial_layer_continuation_stopped",
                    "all_units_committed_layer_manifest_publication_failed",
                    "complete_work_item_only", "partial_work_item_stopped"), "terminal grouped work record required")
        committed = ([span_from_record(row) for row in record["committed_spans"]]
                     if committed_index is None else committed_index.get(ref.span.key, ()))
        require(any(span.key == ref.span.key and span.start <= ref.span.start < ref.span.stop <= span.stop
                    for span in committed), "group rows not in exact committed coverage")
        require(record.get("shared_state_reference") == str(BASE / state.path), "group normalization reference mismatch")
        expected_path = str(PurePosixPath(ref.status_path).parent / "channels.h5")
    require(ref.path == expected_path, "terminal record belongs to another fragment path")
    if recovered_item is not None:
        # This exception is after all ordinary record/role/path/span bindings.
        # It changes only the two historical-publication vetoes above; every
        # physical marker/voxel/weight/bias/coordinate check still runs below.
        require(ref.format == "grouped_channels", "recovery applies only to exact original grouped fragments")
        allowed = recovery.prefix_allowance(recovery_review, recovered_item["work_item_id"], record,
            base_directory=str(BASE if base_directory is None else base_directory))
        approved = [span_from_record(row) for row in allowed]
        require(any(span.key == ref.span.key and span.start <= ref.span.start < ref.span.stop <= span.stop
                    for span in approved), "fragment rows exceed reviewed recovered prefix")


@dataclass
class Normalization:
    participant: str
    native_ids: object
    eligible_ids: object
    eligible_mask: object
    mean: object
    std: object
    training_image_ids: object
    repetition_counts: object


def validate_normalization(arrays, binding, participant):
    """Small state arrays only; never request normalized_X or penalized_Gram."""
    import numpy as np
    native, eligible, mask = (np.asarray(arrays[key]) for key in
        ("all_native_voxel_ids", "eligible_native_voxel_ids", "eligible_mask"))
    mean, std = (np.asarray(arrays[key]) for key in ("neural_mean", "neural_std"))
    image_ids, counts = (np.asarray(arrays[key]) for key in ("training_image_ids", "image_repetition_counts"))
    require(native.ndim == eligible.ndim == 1 and native.dtype.kind in "iu" and eligible.dtype.kind in "iu"
            and len(eligible) >= 500 and bool((native >= 0).all())
            and bool((native[1:] > native[:-1]).all()), "sorted unique native voxel axis required")
    require(mask.shape == native.shape and mask.dtype.kind == "b" and np.array_equal(native[mask], eligible),
            "eligible native voxel binding mismatch")
    require(mean.shape == std.shape == (1, len(native)) and mean.dtype.str == std.dtype.str == "<f4"
            and bool(np.isfinite(mean).all()) and bool(np.isfinite(std).all())
            and bool((std >= 0).all()) and np.array_equal(mask, std[0] > 0), "fitting-only neural moments/eligibility invalid")
    groups = binding.get("preserved_trial_rows_per_image", [])
    require(binding.get("participant") == participant and binding.get("role") == "subject_unique"
            and binding.get("image_mean_weighting") == "equal_images"
            and image_ids.shape == counts.shape == (4712,)
            and counts.dtype.kind in "iu" and bool((counts > 0).all())
            and len(set(image_ids.tolist())) == 4712
            and binding.get("training_image_ids") == image_ids.tolist()
            and len(groups) == 4712 and all(groups)
            and all(type(row) is int for group in groups for row in group)
            and np.array_equal(counts, [len(group) for group in groups])
            and sorted(row for group in groups for row in group) == list(range(int(counts.sum()))),
            "complete unique-image/repetition training binding required")
    return Normalization(participant, native, eligible, mask, mean[0], std[0], image_ids, counts)


def load_normalization(base, state, check):
    """Bound and load only small normalization/identity vectors from the NPZ."""
    import numpy as np
    path = physical_path(base, state.path)
    keys = ("all_native_voxel_ids", "eligible_native_voxel_ids", "eligible_mask",
            "neural_mean", "neural_std", "training_image_ids", "image_repetition_counts")
    total = 0
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {key + ".npy" for key in (*keys, "normalized_X", "penalized_Gram")},
                "exact retained training-state arrays required")
        for key in keys:
            check()
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "supported bounded NPY header required")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                require(not dtype.hasobject and len(shape) in (1, 2), "nonobject identity/moment arrays required")
                size = 1
                for dimension in shape:
                    size *= dimension
                total += size * dtype.itemsize
                require(total <= MAX_NORMALIZATION_BYTES, "normalization vector byte cap exceeded")
    with np.load(path, allow_pickle=False) as arrays:
        small = {key: arrays[key] for key in keys}
    binding = _read_json(physical_path(base, state.binding_path))
    return validate_normalization(small, binding, state.participant)


def _attrs_equal(group, expected):
    for key, value in expected.items():
        require(key in group.attrs and group.attrs[key] == value, f"fragment attribute mismatch: {key}")


def fragment_header(handle, ref):
    """Verify exact existing layout and return storage origin, never infer axes."""
    import h5py
    root = handle
    if ref.format == "retained_prefix":
        _attrs_equal(root, {"schema_version": "ep21.retained_decoder_prefix_layer.v1",
                           "layer": ref.span.layer, "source_fragment_axes": AXES})
        group, origin, limit = root, 0, 64
    elif ref.format == "continuation_fragment":
        _attrs_equal(root, {"schema_version": "ep21.decoder_continuation_fragment.v1"})
        group = root
        origin, limit = int(group.attrs["spatial_Fortran_start"]), int(group.attrs["spatial_Fortran_stop"])
    else:
        require(root.attrs.get("schema_version") in ("ep21.decoder_channel_groups.v1", "ep21.decoder_work_item_fragment.v1"),
                "unsupported grouped decoder schema")
        _attrs_equal(root, {"participant": ref.span.participant})
        # External/soft links are not granted implicit access by a permitted file.
        parent = root
        for segment in ref.group.split("/"):
            require(isinstance(parent.get(segment, getlink=True), h5py.HardLink), "external/soft HDF5 group link forbidden")
            parent = parent[segment]
        group = parent
        origin, limit = int(group.attrs["spatial_Fortran_start"]), int(group.attrs["spatial_Fortran_stop"])
    if ref.format != "retained_prefix":
        _attrs_equal(group, {"participant": ref.span.participant, "layer": ref.span.layer,
                            "channel": ref.span.channel, "source_fragment_axes": AXES})
    require(origin <= ref.span.start < ref.span.stop <= limit, "declared fragment rows exceed stored interval")
    count = limit - origin
    descriptors = {"state": ((count,), "|u1"), "selected_native_voxel_ids": ((count, 500), "<i8"),
                   "weights": ((count, 500), "<f4"), "bias": ((count,), "<f4"),
                   "target_mean": ((count,), "<f4"), "target_std": ((count,), "<f4"),
                   "solve_seconds": ((count,), "<f8"), "full_layer_Fortran_index": ((count,), "<i8")}
    if ref.format == "retained_prefix":
        descriptors["tensor_coordinate"] = ((count, len(ref.span.shape)), "<i8")
    else:
        descriptors.update(spatial_Fortran_index=((count,), "<i8"), full_layer_C_index=((count,), "<i8"))
    require(set(group) == set(descriptors), "exact fragment dataset vocabulary required")
    for name, (shape, dtype) in descriptors.items():
        require(isinstance(group.get(name, getlink=True), h5py.HardLink), "external/soft dataset link forbidden")
        data = group[name]
        require(isinstance(data, h5py.Dataset) and data.shape == shape and data.dtype.str == dtype
                and not data.is_virtual and data.external is None,
                "bounded ordinary fragment dataset required: " + name)
    return group, origin


@dataclass
class SparseDecoderBlock:
    """Logical source W is native_voxel x full_layer_Fortran_unit, with bias separate.

    Storage remains [unit, ranked selected voxel]. native_columns index the full
    sorted native-ID axis; omitted voxels have implicit zero coefficients. Only
    selected (eligible) voxels are normalized, so zero-variance excluded voxels
    are never divided by zero. For unit u the exact conceptual evaluation is:
      normalized_target[u] = sum_k(((raw[id[u,k]]-mean[u,k])/std[u,k]) * W[u,k]) + b[u]
      target[u] = normalized_target[u] * target_std[u] + target_mean[u].
    This object does not execute that evaluation or grant prediction authority.
    Rank order and float32 fitted coefficients/bias are preserved, not refitted.
    """
    participant: str
    layer: str
    full_layer_Fortran_index: object
    full_layer_C_index: object
    selected_native_voxel_ids: object
    native_columns: object
    weights: object
    bias: object
    neural_mean: object
    neural_std: object
    target_mean: object
    target_std: object


def read_sparse_block(group, origin, ref, normalization, begin, end):
    import numpy as np
    require(ref.span.participant == normalization.participant, "cross-participant normalization forbidden")
    require(ref.span.start <= begin < end <= ref.span.stop and end - begin <= MAX_BLOCK_UNITS,
            "bounded in-scope unit block required")
    rows = slice(begin - origin, end - origin)
    require(bool((group["state"][rows] == 2).all()), "every assembled row must be durably complete")
    spatial = np.arange(begin, end, dtype=np.int64)
    shape = ref.span.shape
    if len(shape) == 3:
        channels, height, width = shape
        full_f = ref.span.channel + channels * spatial
        full_c = ref.span.channel * height * width + (spatial % height) * width + spatial // height
        coordinates = np.column_stack((np.full(len(spatial), ref.span.channel), spatial % height, spatial // height))
    else:
        full_f, full_c, coordinates = spatial, spatial, spatial[:, None]
    require(np.array_equal(group["full_layer_Fortran_index"][rows], full_f), "source Fortran target axis mismatch")
    if ref.format == "retained_prefix":
        require(np.array_equal(group["tensor_coordinate"][rows], coordinates), "retained tensor coordinates mismatch")
    else:
        require(np.array_equal(group["spatial_Fortran_index"][rows], spatial)
                and np.array_equal(group["full_layer_C_index"][rows], full_c), "C/F target coordinate mismatch")
    ids = group["selected_native_voxel_ids"][rows]
    require(bool((np.diff(np.sort(ids, axis=1), axis=1) > 0).all()), "duplicate selected native voxel in a unit")
    columns = np.searchsorted(normalization.native_ids, ids)
    require(bool((columns < len(normalization.native_ids)).all()), "unknown selected native voxel")
    require(np.array_equal(normalization.native_ids[columns], ids)
            and bool(normalization.eligible_mask[columns].all()), "unknown or ineligible selected native voxel")
    weights, bias, mean, std, elapsed = (group[key][rows] for key in
        ("weights", "bias", "target_mean", "target_std", "solve_seconds"))
    require(all(bool(np.isfinite(value).all()) for value in (weights, bias, mean, std, elapsed))
            and bool((std > 0).all()) and bool((elapsed >= 0).all()), "invalid fitted coefficient or training moment")
    return SparseDecoderBlock(ref.span.participant, ref.span.layer, full_f, full_c,
        ids, columns, weights, bias, normalization.mean[columns], normalization.std[columns], mean, std)


def _scan_fragments(base, refs, state, normalization, check, block_units, *, recovery_review=None):
    """One HDF5 handle and parsed terminal report per physical fragment file.

    Only one coefficient block is yielded at a time; no layer/all-file weight
    cache is built. A grouped status's span index is small interval metadata.
    """
    import h5py
    ordered = sorted(refs, key=lambda ref: (ref.path, ref.status_path,
                     coverage.prefix.LAYERS.index(ref.span.layer), ref.span.channel, ref.span.start))
    for (path, status_path), items in groupby(ordered, key=lambda ref: (ref.path, ref.status_path)):
        check()
        record = _read_json(physical_path(base, status_path))
        committed_index = {}
        for row in record.get("committed_spans", []):
            span = span_from_record(row)
            committed_index.setdefault(span.key, []).append(span)
        file_refs = tuple(items)
        for ref in file_refs:
            if recovery_review is None:
                validate_terminal_record(ref, record, state, committed_index)
            else:
                validate_terminal_record(ref, record, state, committed_index,
                    recovery_review=recovery_review, base_directory=base)
        with h5py.File(physical_path(base, path), "r") as handle:
            for ref in file_refs:
                check()
                group, origin = fragment_header(handle, ref)
                for begin in range(ref.span.start, ref.span.stop, block_units):
                    check()
                    yield read_sparse_block(group, origin, ref, normalization, begin,
                                            min(begin + block_units, ref.span.stop))


@dataclass(frozen=True)
class CompleteDecoderManifest:
    fragments: tuple[FragmentRef, ...]
    states: tuple[StateRef, ...]
    recovery_review: dict | None = None


def _validate_recovery_catalog(fragments, review):
    """A recovered collection must keep both exact physical pieces and review."""
    if review is None:
        require(not any(ref.path.startswith(FULL_ROOT + "/io_remainder_shards_001/")
                        for ref in fragments), "I/O remainder fragments require their explicit review binding")
        return
    import reviewed_io_recovery as recovery
    items = recovery.validate_review_structure(review)
    for item in items.values():
        for directory, status, rows in (
                (str(PurePosixPath(item["predecessor_status"]).parent), item["predecessor_status"], item["committed_spans"]),
                (item["output_directory"], item["output_directory"] + "/work_item_status.json", item["spans"])):
            expected = [FragmentRef(directory + "/channels.h5", "grouped_channels",
                f"{span.layer}/channel_{span.channel:03d}", status, span)
                for span in map(span_from_record, rows)]
            actual = [ref for ref in fragments if ref.path == directory + "/channels.h5"]
            require(sorted(actual, key=lambda ref: ref.span.key + (ref.span.start,))
                    == sorted(expected, key=lambda ref: ref.span.key + (ref.span.start,)),
                    "complete catalog must preserve exact reviewed prefixes and fresh complements")


def validate_complete_manifest(record):
    """Validate serialized full-scope metadata; its coefficients stay in place."""
    require(record.get("schema_version") == "ep21.assembled_decoder.v1"
            and record.get("status") == "complete_fixed_decoder_collection"
            and record.get("complete_decoder") is True
            and record.get("physical_rows_validated") == record.get("units") == coverage.TOTAL
            and record.get("participants") == list(coverage.PARTICIPANTS)
            and record.get("layers") == list(coverage.prefix.LAYERS)
            and record.get("layer_shapes") == dict(zip(coverage.prefix.LAYERS, map(list, coverage.prefix.SHAPES)))
            and record.get("normalization") == NORMALIZATION and record.get("source_fragment_axes") == AXES,
            "complete physically validated decoder collection required")
    fragments = tuple(FragmentRef.from_record(value) for value in record["fragments"])
    states = tuple(StateRef(**{key: value[key] for key in ("participant", "path", "binding_path", "status_path")})
                   for value in record["normalization_states"])
    verify_catalog(fragments, states)
    # Legacy serialized manifests have no added field and keep their old form.
    review = record.get("reviewed_io_recovery")
    _validate_recovery_catalog(fragments, review)
    return CompleteDecoderManifest(fragments, states, review)


def _require_reader_context(budget):
    require(budget is not None and hasattr(budget, "allocation") and callable(getattr(budget, "check", None)),
            "caller-owned verified compute budget required")
    allocation = budget.allocation
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(isinstance(allocation, dict) and job.isdigit() and "-ln" not in host
            and allocation.get("job_id") == job and allocation.get("hostname") == host,
            "caller allocation must name the actual compute shell")
    end, deadline = allocation.get("job_end_epoch"), getattr(budget, "deadline", None)
    require(isinstance(end, (int, float)) and isinstance(deadline, (int, float))
            and math.isfinite(end) and math.isfinite(deadline)
            and 0 < deadline - time.monotonic() <= end - time.time(),
            "caller compute deadline must remain inside its verified allocation")
    # The caller already verified CPUs/memory/GPUs for its own authorized stage.
    # A future converter need not have the assembler's1CPU/7GiB resource shape.
    # Reuse that proof instead of issuing another scheduler query.
    budget.check()


def load_complete_manifest(path, *, budget):
    """Read the fixed assembly output within the caller's authorized stage."""
    _require_reader_context(budget)
    require(Path(path).absolute() == BASE / MANIFEST, "fixed assembled manifest path required")
    budget.check()
    result = validate_complete_manifest(_read_json(physical_path(BASE, MANIFEST)))
    if result.recovery_review is not None:
        _revalidate_recovery_review(result.recovery_review, budget.check)
    return result


def _publish_new_json(path, record, byte_cap):
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    require(len(payload) <= byte_cap, "assembled metadata exceeds authorized durable cap")
    temporary = path.with_suffix(path.suffix + ".building")
    require(not path.exists() and not temporary.exists(), "assembly output is once-only, never overwrite")
    try:
        with open(temporary, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Link publication cannot replace a raced existing manifest.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _admit_remainder_items(path, authority, work_manifest, allocation, check):
    """Metadata admission inside the already verified assembly allocation.

    Both physical attempts remain in the accounting history. A successful
    registered remainder is required before any coefficient or normalization
    payload is opened; the later row scan still verifies every physical row.
    """
    import decoder_batch_accounting as accounting
    import decoder_deadline_remainders as remainders
    import decoder_remainder_bundle as bundle
    path = Path(path).absolute()
    require(path in (BASE / remainders.PLAN_RELATIVE, BASE / remainders.PLAN_RELATIVE_002),
            "fixed explicit remainder manifest required")
    check()
    plan = _read_json(physical_path(BASE, str(path.relative_to(BASE))))
    require(BASE / remainders.plan_relative(plan) == path, "remainder plan content identity mismatch")
    planned = remainders.validate_plan_structure(plan, authority, work_manifest)
    review = _read_json(physical_path(BASE, plan["reviewed_partials"]))
    with accounting.locked_events(BASE / FULL_ROOT / "submission.jsonl", shared=True) as (events, _):
        account = accounting.replay(events, authority, work_manifest)
    assembly_job = account["jobs"].get(allocation["job_id"])
    require(assembly_job is not None and assembly_job["phase"] == "assembly"
            and assembly_job["logical_id"] == "assembly" and assembly_job["terminal"] is None
            and account["batches"].get(assembly_job["batch_id"], {}).get(remainders.FIELD) == plan,
            "assembly allocation must explicitly register this remainder manifest")
    records = {}
    for work, item in planned.items():
        check()
        records[work] = _read_json(physical_path(BASE, item["predecessor_status"]))
    admitted = remainders.validate_plan(plan, authority, work_manifest, review, records, account,
                                        base_directory=str(BASE))
    for work, item in admitted.items():
        check()
        job = remainders.registered_remainder_job(account, plan, work, require_terminal=True)
        record = _read_json(physical_path(BASE, item["output_directory"] + "/work_item_status.json"))
        registered_plan = account["batches"][job["batch_id"]][remainders.FIELD]
        remainders.validate_completed_remainder(item, record, job, base_directory=str(BASE),
            remainder_plan_relative=remainders.plan_relative(registered_plan),
            registered_work_items=bundle.work_ids_for_job(job, account["batches"]))
    return admitted


def _recovery_account(authority, work_manifest):
    import decoder_batch_accounting as accounting
    with accounting.locked_events(BASE / FULL_ROOT / "submission.jsonl", shared=True) as (events, _):
        return accounting.replay(events, authority, work_manifest)


def _validate_recovery_evidence(review, work_manifest, account, check):
    """Closed operational evidence only; no coefficients or training payloads."""
    import reviewed_io_recovery as recovery
    items = recovery.validate_review_structure(review, work_manifest)
    records, reports = {}, {}
    for work, item in items.items():
        check()
        records[work] = _read_json(physical_path(BASE, item["predecessor_status"]))
        report_path = item["inspection"]["report"]
        if report_path not in reports:
            check()
            reports[report_path] = _read_json(physical_path(BASE, report_path))
    admitted = recovery.validate_review(review, work_manifest, account["jobs"], records, reports,
                                        base_directory=str(BASE))
    for work, item in admitted.items():
        check()
        job = recovery.registered_remainder_job(account, review, work, require_terminal=True)
        record = _read_json(physical_path(BASE, item["output_directory"] + "/work_item_status.json"))
        recovery.validate_completed_remainder(item, record, job, base_directory=str(BASE))
    return admitted


def _admit_recovery_review(path, authority, work_manifest, allocation, check):
    """Require this exact review in the current registered assembly allocation."""
    import reviewed_io_recovery as recovery
    require(Path(path).absolute() == BASE / recovery.REVIEW_RELATIVE,
            "fixed explicit I/O recovery review required")
    check()
    review = _read_json(physical_path(BASE, recovery.REVIEW_RELATIVE))
    account = _recovery_account(authority, work_manifest)
    job = account["jobs"].get(allocation["job_id"])
    require(job is not None and job["phase"] == "assembly" and job["logical_id"] == "assembly"
            and job["terminal"] is None
            and account["batches"].get(job["batch_id"], {}).get(recovery.FIELD) == review,
            "assembly allocation must explicitly register this I/O recovery review")
    _validate_recovery_evidence(review, work_manifest, account, check)
    return review


def _revalidate_recovery_review(review, check):
    """Rebind later sparse reads to unchanged review and completed physical work."""
    import reviewed_io_recovery as recovery
    check()
    require(_read_json(physical_path(BASE, recovery.REVIEW_RELATIVE)) == review,
            "assembled I/O recovery binding differs from the fixed review")
    check()
    authority = _read_json(BASE / "outputs/decoder_training_full_authorization.json")
    work_manifest = _read_json(physical_path(BASE, FULL_ROOT + "/work_manifest.json"))
    account = _recovery_account(authority, work_manifest)
    _validate_recovery_evidence(review, work_manifest, account, check)


def assemble(authorization_path, *, block_units=64, remainder_manifest=None, recovery_review=None):
    """Future full-program assembly; consumes no source responses or prediction data."""
    require(type(block_units) is int and 1 <= block_units <= MAX_BLOCK_UNITS, "bounded assembly block required")
    import participant_training_state as state_builder
    authority = state_builder.authorize_full(authorization_path)
    require(authority.get("allow_decoder_fragment_assembly") is True
            and authority.get("assembled_manifest") == MANIFEST,
            "separate full-scope assembly authority required")
    import decoder_execution_contract as execution
    resources = authority["assembly_resources"]
    allocation = execution.verify_allocation(resources)
    budget = execution.WorkBudget(allocation, resources)
    manifest_path = physical_path(BASE, MANIFEST)
    require(not manifest_path.exists(), "completed assembly cannot be replayed")
    plan = _read_json(BASE / authority["work_manifest"])
    execution.validate_work_manifest(plan, authority)
    execution.admit_registered_allocation(authority, plan, "assembly", allocation)
    retained = execution.read_retained_coverage(plan["retained"])
    remainder_items = (None if remainder_manifest is None else
                       _admit_remainder_items(remainder_manifest, authority, plan, allocation, budget.check))
    review = (None if recovery_review is None else
              _admit_recovery_review(recovery_review, authority, plan, allocation, budget.check))
    fragments, states = planned_catalog(plan, retained, remainder_items=remainder_items, recovery_review=review)
    verified = 0
    state_metadata = []
    for state in states:
        budget.check()
        normalization = _admit_normalization(state, budget.check)
        state_metadata.append({**state.record(), "native_voxels": len(normalization.native_ids),
                               "eligible_voxels": len(normalization.eligible_ids), "training_images": 4712})
        selected = [ref for ref in fragments if ref.span.participant == state.participant]
        scan = (_scan_fragments(BASE, selected, state, normalization, budget.check, block_units)
                if review is None else _scan_fragments(BASE, selected, state, normalization,
                    budget.check, block_units, recovery_review=review))
        for block in scan:
            verified += len(block.bias)
        del normalization
    require(verified == coverage.TOTAL, "every target row must be physically validated")
    record = {"schema_version": "ep21.assembled_decoder.v1", "status": "complete_fixed_decoder_collection",
        "participants": list(coverage.PARTICIPANTS), "layers": list(coverage.prefix.LAYERS),
        "layer_shapes": dict(zip(coverage.prefix.LAYERS, map(list, coverage.prefix.SHAPES))),
        "units": verified, "complete_decoder": True, "physical_rows_validated": verified,
        "fragments": [ref.record() for ref in fragments], "normalization_states": state_metadata,
        "normalization": NORMALIZATION, "source_fragment_axes": AXES,
        "logical_weight_axes": "native_voxel_by_full_layer_Fortran_unit",
        "no_dense_weight_matrix_or_duplicate_coefficients": True,
        "prediction_scoring_converter_or_reconstruction_authorized": False,
        "tau_train_values_read": False, "tau_test_values_read": False, "OOD_access": False,
        "assembly_allocation": allocation, "parent_allocation_charged_once": True}
    if remainder_manifest is not None:
        import decoder_deadline_remainders as remainders
        record["remainder_manifest"] = str(Path(remainder_manifest).absolute().relative_to(BASE))
    if review is not None:
        import reviewed_io_recovery as recovery
        record[recovery.FIELD] = review
    budget.check()
    _publish_new_json(manifest_path, record, resources["durable_bytes"])
    return {"status": record["status"], "units": verified, "manifest": str(manifest_path)}


def iter_sparse_blocks(manifest, *, participant, layer, budget, block_units=64, channel=None):
    """Bounded W/b library interface; caller retains its existing stage authority.

    Blocks carry explicit source F/C target IDs, so the caller scatters by those
    IDs rather than assuming contiguous global target order from file traversal.
    An explicit channel selects only that channel's fragments; None preserves
    the existing whole-layer traversal. FC layers have only channel zero.
    No new fitting, prediction, scoring or converter process is launched here.
    """
    _require_reader_context(budget)
    require(isinstance(manifest, CompleteDecoderManifest), "validated complete manifest required")
    require(participant in coverage.PARTICIPANTS and layer in coverage.prefix.LAYERS
            and type(block_units) is int and 1 <= block_units <= MAX_BLOCK_UNITS,
            "bounded authorized participant/layer scope required")
    shape = coverage.prefix.SHAPES[coverage.prefix.LAYERS.index(layer)]
    require(channel is None or (type(channel) is int
            and 0 <= channel < (shape[0] if len(shape) == 3 else 1)),
            "exact in-range channel or whole-layer None required")
    verify_catalog(manifest.fragments, manifest.states)
    if manifest.recovery_review is not None:
        _validate_recovery_catalog(manifest.fragments, manifest.recovery_review)
        _revalidate_recovery_review(manifest.recovery_review, budget.check)
    else:
        _validate_recovery_catalog(manifest.fragments, None)
    state = next(value for value in manifest.states if value.participant == participant)
    normalization = _admit_normalization(state, budget.check)
    selected = [ref for ref in manifest.fragments if ref.span.participant == participant
                and ref.span.layer == layer and (channel is None or ref.span.channel == channel)]
    if manifest.recovery_review is None:
        yield from _scan_fragments(BASE, selected, state, normalization, budget.check, block_units)
    else:
        yield from _scan_fragments(BASE, selected, state, normalization, budget.check,
                                  block_units, recovery_review=manifest.recovery_review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--block-units", type=int, default=64)
    parser.add_argument("--remainder-manifest",
                        help="Explicit immutable original or cumulative reviewed remainder plan")
    parser.add_argument("--io-recovery-review", dest="recovery_review",
                        help="Explicit fixed review of the five inspected I/O-failed prefixes")
    args = parser.parse_args()
    print(json.dumps(assemble(args.authorization, block_units=args.block_units,
                              remainder_manifest=args.remainder_manifest,
                              recovery_review=args.recovery_review)), flush=True)


if __name__ == "__main__":
    main()
