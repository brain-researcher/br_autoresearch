"""Dormant bounded Finding A collection; no empirical reader, CLI or grant.

The caller admits the supplied models, native axes, common truth and tau-test
identities, and supplies an actual allocation budget and aggregate byte cap.
Before opening and throughout this object's lifetime, the caller must maintain
a finite positive RLIMIT_FSIZE soft limit no larger than that byte cap and
SIGXFSZ's default disposition. The caller owns an isolated collection process:
cap exhaustion terminates that process, and its parent must record a resource
failure with the bounded partial files, never a completed publication. Native
sec2 HDF5 I/O enforces the cap without Python file-object callbacks. Recovery
after native EFBIG is not supported. This library only checks the prerequisite;
it never changes process limits or signals, spawns processes or forces an exit.
The admission callback is an externally enforced prerequisite, not evidence
that this library authenticates data or supplies filesystem isolation.

Decoded float32 values are saved once in bounded, append-ordered unit blocks.
Replay always returns the original 224 rows: score_layer alone applies each
bootstrap draw. The point-only sink preserves full correlations/masks. A final
HDF5 file and two small JSON sidecars are published exclusively, report last.
Failure preserves partial or already-visible files; there is no retry/resume.
Full-scale storage and replay cost have not been qualified by this module.
Paired contrast tables and interval numerical values must come from the
existing inference components. Publication binds compact point reductions;
it does not repeat paired-support tests or bootstrap calculations.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import json
import math
import os
from pathlib import Path
import resource
import signal

import h5py
import numpy as np

import finding_a_prediction as prediction
import finding_a_metrics as metrics
import finding_a_inference as inference
import finding_a_secondary as secondary

assembly = prediction.assembly
PARTICIPANTS, LAYERS = inference.PARTICIPANTS, inference.LAYERS
METHODS = inference.METHODS
STREAMS = tuple((method, source, target)
    for source in PARTICIPANTS
    for method, target in [("within_source", None)] + [
        (method, target) for target in PARTICIPANTS if target != source
        for method in ("content_loss", "brain_loss")])
SLOTS = tuple((method, source, target, layer) for layer in LAYERS
              for method, source, target in STREAMS)
_SLOT_NUMBER = {key: index for index, key in enumerate(SLOTS)}
_IO_BYTES = 1024 * 1024
_VECTOR_UNITS = 65536


@dataclass(frozen=True)
class TruthBlock:
    image_ids: tuple[str, ...]
    layer: str
    full_layer_C_index: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    feature_identity: Mapping


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(values):
    values.setflags(write=False)
    return values


def _json(value):
    """Small report conversion; large vectors are referenced in HDF5."""
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        _require(math.isnan(value), "infinite_report_value")
        return None
    return value


def _identity(value):
    _require(isinstance(value, Mapping), "declared_feature_identity_required")
    record = dict(value)
    _require(record.get("role") == "tau_test" and record.get("dtype") == "float32"
             and record.get("extraction_status") == "complete"
             and all(isinstance(record.get(key), str) and record[key]
                     for key in ("checkpoint", "preprocessing")),
             "fixed_tau_test_feature_extraction_identity_required")
    text = json.dumps(record, sort_keys=True, allow_nan=False)
    _require(len(text.encode("utf-8")) <= 65536, "bounded_feature_identity_required")
    return text


def _require_file_limit(maximum_bytes):
    """Require the caller's fatal cap; never change its limit or signal."""
    soft, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    _require(soft != resource.RLIM_INFINITY and 0 < soft <= maximum_bytes,
             "finite_caller_RLIMIT_FSIZE_at_or_below_collection_cap_required")
    _require(signal.getsignal(signal.SIGXFSZ) == signal.SIG_DFL,
             "caller_SIGXFSZ_default_disposition_required")


class FindingACollection:
    def __init__(self, path, *, image_ids, decoder_axes_by_owner,
                 decoder_training_image_ids_by_owner, fitted_object_ids,
                 feature_identity, budget, maximum_bytes, require_admitted):
        _require(callable(require_admitted), "external_collection_admission_required")
        require_admitted()
        assembly._require_reader_context(budget)
        _require(type(maximum_bytes) is int and maximum_bytes > 0,
                 "explicit_aggregate_collection_byte_cap_required")
        _require_file_limit(maximum_bytes)
        self.path = Path(path)
        _require(self.path.is_absolute() and self.path.suffix == ".h5"
                 and self.path.parent.is_dir(), "existing_absolute_hdf5_destination_required")
        self.manifest_path = self.path.with_suffix(".decoded_feature_manifest.json")
        self.report_path = self.path.with_suffix(".finding_a_report.json")
        self.partial = self.path.with_name(self.path.name + ".partial")
        for final in (self.path, self.manifest_path, self.report_path):
            if os.path.lexists(final) or os.path.lexists(str(final) + ".partial"):
                raise FileExistsError(final)
        self.budget, self.maximum_bytes, self.require_admitted = budget, maximum_bytes, require_admitted
        self._closing = self._failed = self._closed = self._sealed = False
        self._h5 = None
        metrics._ids(image_ids, 224)
        self.image_ids = image_ids
        training, _ = inference._context(decoder_training_image_ids_by_owner, image_ids, None, budget.check)
        self.training = dict(training)
        _require(set(decoder_axes_by_owner) == set(PARTICIPANTS), "five_frozen_decoder_axes_required")
        self.axes = {owner: prediction._axis(decoder_axes_by_owner[owner], budget.check)
                     for owner in PARTICIPANTS}
        _require(all(axis.participant == owner for owner, axis in self.axes.items()),
                 "decoder_axis_owner_mismatch")
        _require(set(fitted_object_ids) == set(STREAMS)
                 and all(value is None or (isinstance(value, str) and value and len(value) <= 4096)
                         for value in fitted_object_ids.values()),
                 "all_45_fitted_object_identities_or_explicit_absences_required")
        self.fitted_object_ids = dict(fitted_object_ids)
        self._feature_identity = _identity(feature_identity)
        self._records = {}
        self._scores = {}
        try:
            _require_file_limit(self.maximum_bytes)
            self._h5 = h5py.File(self.partial, "x", driver="sec2")
            self._h5.attrs["role"] = "tau_test"
            self._h5.attrs["complete"] = False
            self._h5.attrs["feature_identity"] = self._feature_identity
            strings = h5py.string_dtype("utf-8")
            self._h5.create_dataset("image_ids", data=image_ids, dtype=strings)
            for owner, axis in self.axes.items():
                self._check()
                group = self._h5.require_group("decoders/" + owner)
                group.create_dataset("training_image_ids", data=self.training[owner], dtype=strings)
                for name in prediction._AXIS_FIELDS:
                    self._check()
                    group.create_dataset(name, data=getattr(axis, name))
            self._h5.flush()
            self._check()
        except BaseException as error:
            self._abort(error)
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self._abort(value)
        else:
            self.close()

    def _check(self):
        _require(not self._closed and not self._failed, "collection_closed_or_failed")
        _require_file_limit(self.maximum_bytes)
        self.require_admitted()
        assembly._require_reader_context(self.budget)

    @contextmanager
    def _operation(self):
        try:
            self._check()
            yield
        except BaseException as error:
            # Close while the operation's local Dataset/Group objects still
            # exist, before unwinding releases their IDs through HDF5.
            self._abort(error)
            raise

    def _abort(self, error):
        self._failed = True
        try:
            self.close()
        except BaseException as cleanup_error:
            # Do not replace the original cancellation or cap failure. A
            # cleanup failure remains visible and the partial is not promoted.
            error.add_note("collection cleanup failed: " + type(cleanup_error).__name__)

    def _slot(self, method, source, target, layer):
        key = (method, source, target, layer)
        _require(key in _SLOT_NUMBER, "fixed_finding_a_layer_slot_required")
        return key, "layers/%03d" % _SLOT_NUMBER[key]

    def _new_slot(self, key):
        _require(not self._sealed and key not in self._records,
                 "decoded_layer_already_written_or_sealed")

    def _record(self, key, group, status, reason=None):
        method, source, target, layer = key
        owner = metrics._owner(method, source, target)
        record = dict(method=method, source=source, target=target, layer=layer,
            decoder_participant=owner, source_native_axis="decoders/" + source + "/full_native_ids",
            decoder_native_axis="decoders/" + owner, image_ids="image_ids",
            fitted_object_id=self.fitted_object_ids[key[:3]], status=status, reason=reason,
            hdf5_group=group, dtype="float32", logical_shape=[224, metrics._unit_count(layer)],
            logical_axes=["image", "full_layer_C_unit"],
            feature_shape=list(assembly.coverage.prefix.SHAPES[LAYERS.index(layer)]))
        self._records[key] = record
        return record

    def unavailable_layer(self, *, method, source, target, layer, reason):
        with self._operation():
            key, group = self._slot(method, source, target, layer)
            self._new_slot(key)
            _require(isinstance(reason, str) and 0 < len(reason) <= 1024,
                     "explicit_bounded_unavailable_reason_required")
            self._record(key, group, "unavailable", reason)

    def write_layer(self, *, method, source, target, layer, blocks):
        """Consume complete (FeaturePredictionBlock, DecoderBlock) pairs once.

        Physical rows append in incoming block order; explicit F/C coordinate
        vectors retain the scientific tensor axes. No full feature bank is
        allocated. The prediction block is a caller-admitted frozen-model output;
        this writer does not refit or reauthenticate that model.
        """
        with self._operation():
            key, location = self._slot(method, source, target, layer)
            self._new_slot(key)
            owner = metrics._owner(method, source, target)
            _require(self.fitted_object_ids[key[:3]] is not None,
                     "decoded_layer_requires_caller_admitted_fitted_object")
            units = metrics._unit_count(layer)
            seen = np.zeros(units, dtype=bool)
            coordinate = "feature_axes/" + layer + "/full_layer_C_index"
            if coordinate not in self._h5:
                axis = self._h5.create_dataset(coordinate, (units,), dtype="<i8")
                for start in range(0, units, _VECTOR_UNITS):
                    self._check()
                    axis[start:start + _VECTOR_UNITS] = np.arange(start, min(units, start + _VECTOR_UNITS))
            group = self._h5.create_group(location)
            group.attrs["complete"] = False
            data = group.create_dataset("source_file_features", (units, 224), dtype="<f4",
                                        chunks=(min(256, units), 224))
            for name in ("full_layer_C_index", "full_layer_Fortran_index"):
                group.create_dataset(name, (units,), dtype="<i8")
            group.create_dataset("valid_mask", (units, 224), dtype="?", chunks=(min(256, units), 224))
            moment_path = "decoders/" + owner + "/feature_moments/" + layer
            first_owner = moment_path not in self._h5
            moments = self._h5.require_group(moment_path)
            if first_owner:
                moments.attrs["complete"] = False
                for name in ("mean", "std"):
                    moments.create_dataset(name, (units,), dtype="<f4")
            else:
                _require(bool(moments.attrs["complete"]), "complete_frozen_owner_moments_required")
            cursor = 0
            iterator = iter(blocks)
            try:
                for predicted, decoder in iterator:
                    self._check()
                    _require(isinstance(predicted, prediction.FeaturePredictionBlock)
                             and (predicted.method, predicted.source, predicted.target, predicted.layer) == key
                             and predicted.decoder_participant == owner and predicted.image_ids == self.image_ids
                             and decoder.layer == layer and decoder.channel == predicted.channel,
                             "prediction_method_owner_image_layer_binding_required")
                    count, f_ids, c_ids = prediction._decoder_block(decoder, self.axes[owner], self.budget.check)
                    _require(np.array_equal(predicted.full_layer_C_index, c_ids)
                             and np.array_equal(predicted.full_layer_Fortran_index, f_ids)
                             and not bool(seen[c_ids].any()), "exact_unique_prediction_F_C_coordinates_required")
                    values = predicted.source_file_features
                    metrics._float32(values, (224, count), "source_file_float32_prediction_rows_required")
                    _require(bool(np.isfinite(values).all()), "nonfinite_decoded_source_file_features")
                    order = np.argsort(c_ids)
                    sorted_ids = c_ids[order]
                    for name, supplied in (("mean", decoder.target_mean), ("std", decoder.target_std)):
                        if first_owner:
                            moments[name][sorted_ids] = supplied[order]
                        else:
                            _require(np.array_equal(moments[name][sorted_ids], supplied[order]),
                                     "frozen_decoder_feature_moments_changed_between_methods")
                    stop = cursor + count
                    data[cursor:stop] = values.T
                    group["valid_mask"][cursor:stop] = np.ones((count, 224), dtype=bool)
                    group["full_layer_C_index"][cursor:stop] = c_ids
                    group["full_layer_Fortran_index"][cursor:stop] = f_ids
                    cursor = stop
                    seen[c_ids] = True
                _require(cursor == units and bool(seen.all()), "complete_all_channel_decoded_layer_required")
                moments.attrs["complete"] = True
                group.attrs["complete"] = True
                self._h5.flush()
                self._check()
                record = self._record(key, location, "decoded")
                record.update(stored_shape=[units, 224], storage_axes=["append_order_unit", "image"],
                    feature_moments=moment_path, coordinates={name: location + "/" + name for name in
                    ("full_layer_C_index", "full_layer_Fortran_index")},
                    values=location + "/source_file_features", valid_mask=location + "/valid_mask")
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

    def seal_decoded(self):
        with self._operation():
            _require(set(self._records) == set(SLOTS), "all_855_decoded_or_unavailable_slots_required")
            self._sealed = True
            self._h5.flush()

    def block_provider(self, *, method, source, target, layer, truth_provider):
        """Replay original rows; the existing scorer alone applies sample indices.

        truth_provider receives layer/image_ids/C indices/feature_identity and
        returns TruthBlock. Its external admission and any underlying source
        readers remain the caller's responsibility. No test-selected filtering.
        """
        self._check()
        _require(self._sealed and callable(truth_provider), "sealed_collection_and_admitted_truth_provider_required")
        key, _ = self._slot(method, source, target, layer)
        record = self._records[key]
        if record["status"] == "unavailable":
            return inference.UnavailableLayer(record["reason"])

        def replay():
            with self._operation():
                group = self._h5[record["hdf5_group"]]
                moments = self._h5[record["feature_moments"]]
                for start in range(0, record["logical_shape"][1], assembly.MAX_BLOCK_UNITS):
                    self._check()
                    stop = min(record["logical_shape"][1], start + assembly.MAX_BLOCK_UNITS)
                    ids = _readonly(group["full_layer_C_index"][start:stop])
                    values = _readonly(group["source_file_features"][start:stop].T.copy())
                    truth = truth_provider(layer=layer, image_ids=self.image_ids,
                        full_layer_C_index=ids, feature_identity=json.loads(self._feature_identity))
                    _require(isinstance(truth, TruthBlock) and truth.layer == layer
                             and truth.image_ids == self.image_ids
                             and np.array_equal(truth.full_layer_C_index, ids)
                             and _identity(truth.feature_identity) == self._feature_identity,
                             "exact_common_truth_image_C_and_feature_identity_required")
                    metrics._float32(truth.values, values.shape, "paired_float32_true_feature_rows_required")
                    _require(isinstance(truth.valid_mask, np.ndarray) and truth.valid_mask.dtype == np.dtype("bool")
                             and truth.valid_mask.shape == values.shape
                             and np.array_equal(truth.valid_mask, np.isfinite(truth.values)),
                             "truth_valid_mask_must_preserve_exact_finite_support")
                    order = np.argsort(ids)
                    inverse = np.argsort(order)
                    mean = _readonly(moments["mean"][ids[order]][inverse])
                    std = _readonly(moments["std"][ids[order]][inverse])
                    self._check()
                    yield metrics.MetricBlock(record["decoder_participant"], self.image_ids, ids,
                                              values, truth.values, mean, std)
        return replay()

    def result_sink(self, score):
        """Retain point correlations/masks with original image and C-unit axes."""
        with self._operation():
            _require(self._sealed and isinstance(score, metrics.LayerScores), "sealed_point_layer_score_required")
            key, location = self._slot(score.method, score.source, score.target, score.layer)
            record = self._records[key]
            _require(record["status"] == "decoded" and key not in self._scores
                     and score.decoder_participant == record["decoder_participant"]
                     and score.image_ids == self.image_ids
                     and score.decoder_training_image_ids == self.training[score.decoder_participant]
                     and np.array_equal(score.image_indices, np.arange(224)),
                     "unique_original_point_score_and_decoder_binding_required")
            units = record["logical_shape"][1]
            moments = self._h5[record["feature_moments"]]
            for name, value in (("mean", score.training_mean), ("std", score.training_std)):
                metrics._float32(value, (units,), "point_frozen_feature_moments_required")
                for start in range(0, units, _VECTOR_UNITS):
                    self._check()
                    _require(np.array_equal(value[start:start + _VECTOR_UNITS],
                                            moments[name][start:start + _VECTOR_UNITS]),
                             "point_score_frozen_moments_disagree")
            summary = {}
            for metric, count in (("pattern", 224), ("profile", units)):
                value = getattr(score, metric)
                _require(isinstance(value, metrics.CorrelationSummary)
                         and value.correlations.dtype == np.dtype("float64")
                         and value.correlations.shape == (count,)
                         and value.defined_mask.dtype == np.dtype("bool")
                         and value.defined_mask.shape == (count,)
                         and np.array_equal(np.isfinite(value.correlations), value.defined_mask)
                         and not bool(np.isinf(value.correlations).any()),
                         "complete_point_correlation_vector_and_defined_mask_required")
                expected = metrics._summary(value.correlations, self.budget.check)
                _require(value.mean == expected.mean and value.defined_count == expected.defined_count
                         and value.undefined_count == expected.undefined_count,
                         "point_correlation_summary_disagrees_with_vector")
                group = self._h5.create_group(location + "/" + metric)
                for name in ("correlations", "defined_mask"):
                    array = getattr(value, name)
                    output = group.create_dataset(name, shape=array.shape, dtype=array.dtype)
                    for start in range(0, count, _VECTOR_UNITS):
                        self._check()
                        output[start:start + _VECTOR_UNITS] = array[start:start + _VECTOR_UNITS]
                summary[metric] = dict(mean=value.mean, defined_count=value.defined_count,
                    undefined_count=value.undefined_count, correlations=group.name + "/correlations",
                    defined_mask=group.name + "/defined_mask",
                    coordinate="image_ids" if metric == "pattern" else
                        "feature_axes/" + score.layer + "/full_layer_C_index")
            self._scores[key] = summary
            self._h5.flush()

    def _bind_results(self, primary, incoming, dyadic):
        _require(isinstance(primary, inference.BootstrapResult)
                 and isinstance(incoming, secondary.IncomingResult)
                 and isinstance(dyadic, secondary.DyadicBootstrapResult), "existing_finding_a_result_types_required")
        panel = primary.point
        secondary._point_tables(panel, self.budget.check)
        for result in (panel, incoming, dyadic):
            _require(result.image_ids == self.image_ids and result.participants == PARTICIPANTS
                     and result.layers == LAYERS, "one_original_panel_identity_required")
        _require(np.array_equal(primary.draw_indices, np.random.RandomState(42).randint(0, 224, (1000, 224)))
                 and dyadic.pair_order == secondary.PAIR_ORDER,
                 "fixed_image_bootstrap_and_dyadic_pair_coordinates_required")
        for metric in ("pattern", "profile"):
            point_metric = getattr(panel, metric)
            reduced = inference._reduce((point_metric.layer_means,
                point_metric.layer_defined_counts, point_metric.layer_undefined_counts,
                point_metric.layer_reasons), self.budget.check)
            _require(all(_json(getattr(point_metric, name)) == _json(getattr(reduced, name))
                         for name in ("layer_source_means", "layer_group_means", "dyad_means",
                                      "source_means", "group_means")),
                     "primary_derived_summary_not_bound_to_layer_tables")
            intervals = getattr(primary, metric)
            _require(isinstance(intervals, inference.MetricIntervals), "complete_primary_interval_tables_required")
            for interval_name, panel_name in inference._INTERVAL_FIELDS:
                estimates = getattr(getattr(panel, metric), panel_name)
                saved = getattr(intervals, interval_name)
                _require(set(saved) == set(estimates), "complete_primary_interval_endpoints_required")
                for endpoint, estimate in estimates.items():
                    interval = saved[endpoint]
                    shape = np.asarray(estimate).shape
                    _require(isinstance(interval, inference.Interval)
                             and all(isinstance(getattr(interval, name), np.ndarray)
                                     and getattr(interval, name).shape == shape
                                     for name in ("lower", "upper", "available", "valid_draws"))
                             and interval.available.dtype == np.dtype("bool")
                             and interval.valid_draws.dtype.kind in "iu"
                             and bool((interval.valid_draws <= 1000).all())
                             and bool((interval.valid_draws >= 0).all()),
                             "bounded_primary_interval_fields_required")
                    expected_available = np.isfinite(inference._endpoint_array(estimate)) & (interval.valid_draws == 1000)
                    _require(np.array_equal(interval.available, expected_available)
                             and np.array_equal(np.isfinite(interval.lower), interval.available)
                             and np.array_equal(np.isfinite(interval.upper), interval.available)
                             and not bool(np.isinf(interval.lower).any())
                             and not bool(np.isinf(interval.upper).any())
                             and bool((interval.lower[interval.available] <= interval.upper[interval.available]).all()),
                             "primary_interval_missingness_or_bounds_disagree")
        _require(dyadic.draw_indices.shape == (1000, 20) and dyadic.draw_indices.dtype.kind in "iu"
                 and dyadic.proposal_counts.shape == (1000,) and dyadic.proposal_counts.dtype.kind in "iu"
                 and dyadic.draw_available.shape == (1000,) and dyadic.draw_available.dtype == np.dtype("bool")
                 and bool((dyadic.draw_indices >= -1).all()) and bool((dyadic.draw_indices < 20).all())
                 and bool((dyadic.proposal_counts <= 1000).all()) and bool((dyadic.proposal_counts >= 20).all())
                 and np.array_equal((dyadic.draw_indices >= 0).all(axis=1), dyadic.draw_available),
                 "fixed_bounded_secondary_draw_fields_required")
        for key, record in self._records.items():
            self._check()
            method, source, target, layer = key
            position = (LAYERS.index(layer), PARTICIPANTS.index(source))
            if target is not None:
                position += (PARTICIPANTS.index(target),)
            for metric in ("pattern", "profile"):
                values = getattr(panel, metric)
                actual = values.layer_means[method][position]
                if record["status"] == "unavailable":
                    _require(np.isnan(actual) and values.layer_reasons[method][position] == record["reason"],
                             "unavailable_collection_slot_not_preserved_in_point_panel")
                else:
                    _require(key in self._scores, "full_point_vectors_required_before_publication")
                    saved = self._scores[key][metric]
                    _require((np.isnan(actual) if saved["mean"] is None else actual == saved["mean"])
                             and values.layer_defined_counts[method][position] == saved["defined_count"]
                             and values.layer_undefined_counts[method][position] == saved["undefined_count"],
                             "point_panel_disagrees_with_saved_point_vectors")
        # Reuse the existing compact secondary reduction, never derive a new
        # contrast by subtracting unmatched method means.
        expected = secondary.target_incoming(panel, check=self.budget.check)
        _require(_json(incoming) == _json(expected), "incoming_summary_not_bound_to_point_panel")
        tables = secondary._point_tables(panel, self.budget.check)
        for metric in ("pattern", "profile"):
            for endpoint in inference.ENDPOINTS:
                table = tables[metric][endpoint]
                rows = np.asarray([table[:, PARTICIPANTS.index(source)] if endpoint == "within_source"
                    else table[:, PARTICIPANTS.index(source), PARTICIPANTS.index(target)]
                    for source, target in secondary.PAIR_ORDER]).T
                _require(np.array_equal(getattr(dyadic, metric).pair_values[endpoint], rows, equal_nan=True),
                         "dyadic_summary_not_bound_to_original_point_table")
                layer_means = np.asarray([secondary._mean_required(row) for row in rows])
                group_mean = secondary._as_group(secondary._mean_required(layer_means))
                _require(np.array_equal(getattr(dyadic, metric).layer_means[endpoint],
                                        layer_means, equal_nan=True)
                         and getattr(dyadic, metric).group_means[endpoint] == group_mean,
                         "dyadic_derived_summary_not_bound_to_original_point_table")

    def _write_array(self, path, array):
        values = np.asarray(array)
        dataset = self._h5.create_dataset(path, shape=values.shape, dtype=values.dtype)
        for start in range(0, len(values), 64):
            self._check()
            dataset[start:start + 64] = values[start:start + 64]

    def _sidecar(self, path, record, used):
        _require_file_limit(self.maximum_bytes)
        partial = path.with_name(path.name + ".partial")
        with partial.open("xb") as stream:
            buffer = bytearray()
            for text in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(_json(record)):
                data = text.encode("utf-8")
                used += len(data)
                _require(used <= self.maximum_bytes, "collection_exceeds_caller_aggregate_byte_cap")
                buffer.extend(data)
                if len(buffer) >= _IO_BYTES:
                    _require_file_limit(self.maximum_bytes)
                    self.require_admitted()
                    self.budget.check()
                    stream.write(buffer)
                    buffer.clear()
            _require_file_limit(self.maximum_bytes)
            self.require_admitted()
            self.budget.check()
            stream.write(buffer)
            stream.flush()
            os.fsync(stream.fileno())
        _require_file_limit(self.maximum_bytes)
        os.link(partial, path)
        partial.unlink()
        return used

    def publish(self, primary, incoming, dyadic):
        """Publish complete requested records, with explicit scientific missingness.

        Missing models/undefined contrasts do not become a complete quantitative
        finding. The final report is last; earlier publication failures leave
        their visible/partial artifacts for the coordinator, without overwrite.
        """
        with self._operation():
            _require(self._sealed, "sealed_complete_roster_required")
            self._bind_results(primary, incoming, dyadic)
            self._write_array("resampling/image_draw_indices", primary.draw_indices)
            for name in ("draw_indices", "proposal_counts", "draw_available"):
                self._write_array("resampling/dyadic_" + name, getattr(dyadic, name))
            manifest = dict(hdf5=str(self.path), image_ids="image_ids", image_count=224,
                role="tau_test", feature_identity=json.loads(self._feature_identity),
                slots=[self._records[key] for key in SLOTS],
                decoded_slots=len(self._scores), unavailable_slots=sum(
                    value["status"] == "unavailable" for value in self._records.values()),
                roster_complete=True, source_file_dtype="float32", true_image_optimization_input=False)
            complete = all(record["status"] == "decoded" for record in self._records.values())
            directional = {}
            for metric in ("pattern", "profile"):
                value = getattr(primary.point, metric).group_means["content_minus_brain"]
                directional[metric] = ("not_evaluable" if value is None else
                                       "positive_estimate" if value > 0 else "nonpositive_estimate")
                complete &= value is not None
                complete &= all(bool(value.available) for value in getattr(primary, metric).group.values())
            report = dict(status="complete_requested_record", finding="A",
                quantitative_finding="estimates_available" if complete else "not_evaluable",
                directional_contrast=directional,
                comparability=dict(status="not_evaluable", reason="missing_outcome_independent_margin",
                                   margin=None, test_run=False),
                primary=dict(point=_json(primary.point), pattern=_json(primary.pattern),
                    profile=_json(primary.profile), image_draw_indices="resampling/image_draw_indices"),
                target_incoming=_json(incoming),
                secondary_dyadic=dict(pair_order=dyadic.pair_order, pattern=_json(dyadic.pattern),
                    profile=_json(dyadic.profile), draw_indices="resampling/dyadic_draw_indices",
                    proposal_counts="resampling/dyadic_proposal_counts", draw_available="resampling/dyadic_draw_available"),
                hdf5=str(self.path), decoded_feature_manifest=str(self.manifest_path),
                point_scores=[dict(slot=self._records[key], **self._scores[key])
                              for key in SLOTS if key in self._scores],
                supplied_component_results=dict(
                    paired_contrast_layer_tables="existing_finding_a_inference_not_recomputed_by_collection",
                    bootstrap_interval_values="existing_primary_and_secondary_components_not_recomputed_by_collection",
                    compact_point_reductions="checked_against_supplied_layer_tables"),
                limitations=["adapted_reproduction", "fixed_five_participants",
                    "descriptive_pointwise_stimulus_intervals_not_population_inference",
                    "twenty_dyads_are_correlated_repeated_targets", "no_comparability_margin",
                    "regular_evidence_is_retrospective_and_correlated_with_ep04",
                    "brain_loss_uses_shared_training_stimuli",
                    "no_OOD_access_or_inference", "full_scale_storage_and_runtime_not_qualified_by_component_tests"])
            self._h5.attrs["complete"] = True
            self._h5.flush()
            self._check()
            self.close()
            _require_file_limit(self.maximum_bytes)
            _require(self.partial.stat().st_size <= self.maximum_bytes,
                     "collection_exceeds_caller_byte_cap")
            os.link(self.partial, self.path)
            self.partial.unlink()
            used = self.path.stat().st_size
            used = self._sidecar(self.manifest_path, manifest, used)
            self._sidecar(self.report_path, report, used)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return self.path, self.manifest_path, self.report_path

    def close(self):
        if self._closed:
            return
        self._closing = True
        descriptor = None
        try:
            _require_file_limit(self.maximum_bytes)
            if self._h5 is not None:
                # Duplicate the actual sec2 descriptor, keeping this same file
                # available to fsync AFTER HDF5 has flushed and closed it.
                # No Python I/O callbacks or alternate backing stream exist.
                descriptor = os.dup(self._h5.id.get_vfd_handle())
                self._h5.close()
                _require_file_limit(self.maximum_bytes)
                os.fsync(descriptor)
        except BaseException:
            self._failed = True
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            # A failed native close never becomes publishable. Preserve the
            # original operation error via _abort. Cap exhaustion instead
            # terminates the caller's isolated process with SIGXFSZ; its parent
            # must retain that resource failure and the bounded partial file.
            if self._h5 is None or not self._h5.id.valid:
                self._closed = True
