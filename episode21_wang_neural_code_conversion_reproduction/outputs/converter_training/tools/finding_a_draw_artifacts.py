"""Bounded point/draw JSON persistence for an admitted Finding A stage.

Only original primitive layer tables are serialized. Derived summaries are
checked before writing and rebuilt with the unchanged reducer after reading;
NaN layer values become JSON null, and undefined group estimates remain None.
No pickle, payload hashing, provider access, scoring, fitting or publication
of a scientific result occurs here. Public panel helpers represent the point;
draw files contain one exact canonical slice of the original 1,000 draws.

The caller authenticates the published stage and its fitted/truth identities,
supplies its immutable 224-image tuple, and admits these local paths/resources.
Exact identity equality is a join to that supplied authority, not independent
authentication of files or scientific values. Interrupted files remain partial;
existing final or partial outputs are never replaced or resumed.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path

import numpy as np

import finding_a_collection as collection
import finding_a_draw_batches as batches
import finding_a_inference as original


_require = original._require
_CHUNK = 65536
_FIELDS = ("layer_means", "layer_defined_counts", "layer_undefined_counts", "layer_reasons")
_PANEL_KEYS = ("image_ids", "image_indices", "participants", "layers", "pattern", "profile")
_IDENTITY_KEYS = ("collection_path", "point_path", "fitted_object_ids", "feature_identity")


def _keys(value, expected, reason):
    _require(isinstance(value, Mapping) and set(value) == set(expected), reason)


def stage_identity_record(value, *, check):
    """Snapshot the four agreed stage fields before admission callbacks.

    The point path is the published decoded-point metadata path plus '#point'.
    Forty-five fitted entries use the original collection.STREAMS order; an
    absent fitted object is represented by None, not a substitute model.
    """
    result = collection._json(value)
    check()
    _keys(result, _IDENTITY_KEYS, "exact_published_stage_identity_fields_required")
    path, point = result["collection_path"], result["point_path"]
    _require(isinstance(path, str) and Path(path).is_absolute() and Path(path).suffix == ".h5"
             and isinstance(point, str) and point.endswith("#point")
             and Path(point[:-6]).is_absolute() and Path(point[:-6]).suffix == ".json",
             "absolute_published_collection_and_point_identity_required")
    rows = result["fitted_object_ids"]
    _require(isinstance(rows, list) and len(rows) == len(collection.STREAMS),
             "all_45_original_fitted_identities_required")
    for row, stream in zip(rows, collection.STREAMS):
        check()
        _keys(row, ("method", "source", "target", "fitted_object_id"),
              "exact_original_fitted_identity_fields_required")
        _require((row["method"], row["source"], row["target"]) == stream,
                 "canonical_45_fitted_identity_order_required")
        fitted = row["fitted_object_id"]
        _require(fitted is None or (isinstance(fitted, str) and 0 < len(fitted) <= 4096),
                 "bounded_original_fitted_object_identity_required")
    result["feature_identity"] = json.loads(collection._identity(result["feature_identity"]))
    check()
    return result


def _panel_record(panel, image_ids, indices, check):
    original._image_ids(image_ids, 224)
    batches._panel(panel, image_ids, indices, check)
    record = {"image_ids": list(panel.image_ids), "image_indices": panel.image_indices.tolist(),
              "participants": list(panel.participants), "layers": list(panel.layers)}
    for name in original._METRICS:
        check()
        metric = getattr(panel, name)
        record[name] = {field: collection._json(getattr(metric, field)) for field in _FIELDS}
    check()
    return record


def panel_to_record(point, *, check):
    """Encode an original point PanelResult; bootstrap rows use batch files."""
    check()
    _require(isinstance(point, original.PanelResult), "original_point_panel_required")
    return _panel_record(point, point.image_ids, np.arange(224, dtype=np.int64), check)


def _array(value, shape, kind):
    _require(isinstance(value, list), "JSON_array_required")
    raw = np.asarray(value, dtype=object)
    _require(raw.shape == shape, "exact_primitive_layer_table_shape_required")
    if kind == "means":
        _require(all(item is None or (type(item) in (int, float) and math.isfinite(item))
                     for item in raw.flat), "finite_or_null_float64_layer_values_required")
        result = np.asarray(value, dtype=np.float64)
    elif kind == "counts":
        _require(all(type(item) is int and -1 <= item <= np.iinfo(np.int64).max for item in raw.flat),
                 "exact_int64_layer_support_counts_required")
        result = np.asarray(value, dtype=np.int64)
    else:
        _require(all(isinstance(item, str) and bool(item) for item in raw.flat),
                 "explicit_layer_reason_strings_required")
        result = raw
    return original._readonly(result)


def _metric_from_record(value, check):
    _keys(value, _FIELDS, "exact_primitive_metric_fields_required")
    for field in _FIELDS:
        _keys(value[field], original.ENDPOINTS if field in ("layer_means", "layer_reasons")
              else original.METHODS, "all_original_metric_endpoints_required")
    means, defined, undefined, reasons = {}, {}, {}, {}
    for endpoint in original.ENDPOINTS:
        check()
        shape = (19, 5) if endpoint == "within_source" else (19, 5, 5)
        means[endpoint] = _array(value["layer_means"][endpoint], shape, "means")
        reasons[endpoint] = _array(value["layer_reasons"][endpoint], shape, "reasons")
        if endpoint in original.METHODS:
            defined[endpoint] = _array(value["layer_defined_counts"][endpoint], shape, "counts")
            undefined[endpoint] = _array(value["layer_undefined_counts"][endpoint], shape, "counts")
    # Paired contrast tables are retained as supplied; never derive a contrast
    # by subtracting method means whose finite support may differ.
    metric = original._reduce((means, defined, undefined, reasons), check)
    batches._metric(metric, check)
    return metric


def _panel_from_record(record, image_ids, indices, check):
    check()
    _keys(record, _PANEL_KEYS, "exact_original_panel_record_fields_required")
    original._image_ids(image_ids, 224)
    _require(record["image_ids"] == list(image_ids)
             and record["participants"] == list(original.PARTICIPANTS)
             and record["layers"] == list(original.LAYERS), "original_panel_identity_mismatch")
    rows = record["image_indices"]
    _require(isinstance(rows, list) and len(rows) == 224
             and all(type(item) is int and 0 <= item < 224 for item in rows),
             "exact_original_integer_image_draw_required")
    rows = original._readonly(np.asarray(rows, dtype=np.int64))
    _require(np.array_equal(rows, indices), "canonical_panel_draw_order_required")
    pattern = _metric_from_record(record["pattern"], check)
    profile = _metric_from_record(record["profile"], check)
    panel = original.PanelResult(image_ids, rows, original.PARTICIPANTS, original.LAYERS, pattern, profile)
    batches._panel(panel, image_ids, indices, check)
    return panel


def panel_from_record(record, *, expected_image_ids, check):
    """Restore the canonical point, including exact original missingness."""
    return _panel_from_record(record, expected_image_ids, np.arange(224, dtype=np.int64), check)


def _file_context(path, maximum_bytes, check):
    _require(callable(check), "external_draw_artifact_check_required")
    _require(type(maximum_bytes) is int and maximum_bytes > 0, "explicit_finite_draw_artifact_cap_required")
    check()
    path = Path(path)
    _require(path.is_absolute() and path.suffix == ".json" and path.resolve() == path,
             "absolute_nonsymlink_draw_JSON_path_required")
    return path


def _expected(expected_stage_identity, expected_batch_index, expected_image_ids, check):
    # Detach the caller's identity before the first callback can change it.
    identity = stage_identity_record(expected_stage_identity, check=check)
    original._image_ids(expected_image_ids, 224)
    start = batches._batch_start(expected_batch_index)
    draws = batches._schedule()[start:start + batches.DRAW_BATCH_SIZE]
    return identity, draws


def _sync_directory(path):
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_draw_batch(path, batch, *, expected_stage_identity, expected_batch_index,
                     expected_image_ids, maximum_bytes, check):
    """Write exactly one admitted original batch, exclusively and within cap.

    A failed write preserves its partial and closes its stream. A failure after
    linking may leave a final name visible; it is not retried or overwritten.
    The caller records success only after this function returns.
    """
    identity, draws = _expected(expected_stage_identity, expected_batch_index, expected_image_ids, check)
    path = _file_context(path, maximum_bytes, check)
    _require(isinstance(batch, batches.DrawBatch) and type(batch.batch_index) is int
             and batch.batch_index == expected_batch_index
             and isinstance(batch.panels, tuple) and len(batch.panels) == len(draws),
             "complete_expected_canonical_draw_batch_required")
    if os.path.lexists(path):
        raise FileExistsError(path)
    record = {"stage_identity": identity, "batch_index": expected_batch_index,
              "panels": [_panel_record(panel, expected_image_ids, indices, check)
                         for panel, indices in zip(batch.panels, draws)]}
    partial = path.with_name(path.name + ".partial")
    used, buffer = 0, bytearray()
    check()
    with partial.open("xb") as stream:
        for token in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(record):
            data = token.encode("utf-8")
            used += len(data)
            _require(used <= maximum_bytes, "draw_artifact_exceeds_caller_byte_cap")
            buffer.extend(data)
            if len(buffer) >= _CHUNK:
                check()
                stream.write(buffer)
                buffer.clear()
        check()
        stream.write(buffer)
        stream.flush()
        os.fsync(stream.fileno())
    check()
    os.link(partial, path)
    _sync_directory(path)
    partial.unlink()
    _sync_directory(path)
    return path


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_draw_JSON_field")
        result[key] = value
    return result


def _constant(unused):
    raise ValueError("nonstandard_nonfinite_JSON_number_forbidden")


def read_draw_batch(path, *, expected_stage_identity, expected_batch_index,
                    expected_image_ids, maximum_bytes, check):
    """Read one bounded published batch from the same caller-admitted stage."""
    identity, draws = _expected(expected_stage_identity, expected_batch_index, expected_image_ids, check)
    path = _file_context(path, maximum_bytes, check)
    check()
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size <= maximum_bytes,
                 "draw_artifact_exceeds_caller_byte_cap")
        raw = bytearray()
        while True:
            check()
            data = stream.read(min(_CHUNK, maximum_bytes + 1 - len(raw)))
            if not data:
                break
            raw.extend(data)
            _require(len(raw) <= maximum_bytes, "draw_artifact_exceeds_caller_byte_cap")
    check()
    record = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    del raw
    _keys(record, ("stage_identity", "batch_index", "panels"), "exact_draw_artifact_fields_required")
    supplied = stage_identity_record(record["stage_identity"], check=check)
    _require(supplied == identity, "same_admitted_collection_point_fitted_and_truth_identity_required")
    _require(type(record["batch_index"]) is int and record["batch_index"] == expected_batch_index,
             "expected_canonical_draw_batch_index_required")
    _require(isinstance(record["panels"], list) and len(record["panels"]) == len(draws),
             "complete_canonical_draw_batch_required")
    panels = tuple(_panel_from_record(value, expected_image_ids, indices, check)
                   for value, indices in zip(record["panels"], draws))
    check()
    return batches.DrawBatch(expected_batch_index, panels)
