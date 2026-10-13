"""Dormant owner/layer writes into the existing Finding A collection format.

An externally admitted mapped decoder block is read once, then passed to the
unchanged predictor separately for every present fitted stream owning that
decoder. Each call retains its original 64-row arithmetic and float32 output;
no rows or model inputs are concatenated. Only one prediction block is retained
at a time. The constructor, native fatal file cap, complete855-slot contract,
point-vector sink, replay and exclusive publication are inherited unchanged.

The caller supplies the original PreparedDecoderInput objects and coefficient
iterator; this module loads no model, grants no access and starts no process.
Same fitted-object and coefficient identities remain caller prerequisites.
Full-scale storage (roughly630GB for37 streams) and runtime are not qualified
by these component checks, and no full-readout resource grant is supplied.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np

import finding_a_collection as original

prediction, metrics = original.prediction, original.metrics


@contextmanager
def _decoder_source(blocks):
    iterator = None
    closed = False

    def begin():
        nonlocal iterator
        iterator = iter(blocks)
        return iterator

    def finish():
        nonlocal closed
        if not closed:
            closed = True
            close = getattr(iterator if iterator is not None else blocks, "close", None)
            if callable(close):
                close()

    try:
        yield begin, finish
    except BaseException as error:
        try:
            finish()
        except BaseException as cleanup_error:
            error.add_note("grouped decoder cleanup failed: " + type(cleanup_error).__name__)
        raise
    else:
        finish()


class GroupedFindingACollection(original.FindingACollection):
    def write_owner_layer(self, *, owner, layer, prepared_by_stream, blocks):
        """Write every present stream for one owner/layer in one decoder pass.

        ``blocks`` yields immutable mapped DecoderBlocks, each containing at
        most256 units. Mapping keys are (method,source,target), exactly matching
        this owner's non-None fitted-object slots. Missing streams stay explicit
        through the inherited unavailable_layer method; they cannot be silently
        omitted from a group whose fitted-object identity is present.
        """
        # Even an admission/identity failure before iter(blocks) closes a
        # supplied iterator. Cleanup never replaces the original failure.
        with _decoder_source(blocks) as (begin, finish):
            with self._operation():
                original._require(owner in original.PARTICIPANTS and layer in original.LAYERS,
                                  "fixed_decoder_owner_and_layer_required")
                streams = tuple(stream for stream in original.STREAMS
                    if metrics._owner(*stream) == owner and self.fitted_object_ids[stream] is not None)
                original._require(streams and isinstance(prepared_by_stream, Mapping)
                    and set(prepared_by_stream) == set(streams),
                    "all_present_owner_streams_required")
                prepared = tuple((stream, prepared_by_stream[stream]) for stream in streams)
                slots = []
                # Validate the entire roster before coefficient iteration,
                # prediction or any new dataset creation.
                for stream, value in prepared:
                    self._check()
                    key, location = self._slot(*stream, layer)
                    self._new_slot(key)
                    original._require(isinstance(value, prediction.PreparedDecoderInput)
                        and (value.method, value.source, value.target) == stream
                        and value.image_ids == self.image_ids,
                        "prepared_stream_and_image_identity_required")
                    prediction._same_axis(value.decoder_axis, self.axes[owner])
                    metrics._float32(value.values, (224, len(self.axes[owner].eligible_native_ids)),
                                     "complete_prepared_float32_rows_required")
                    original._require(not value.values.flags.writeable,
                                      "immutable_prepared_decoder_rows_required")
                    slots.append((key, location))

                units = metrics._unit_count(layer)
                seen = np.zeros(units, dtype=bool)
                coordinate = "feature_axes/" + layer + "/full_layer_C_index"
                if coordinate not in self._h5:
                    axis = self._h5.create_dataset(coordinate, (units,), dtype="<i8")
                    for start in range(0, units, original._VECTOR_UNITS):
                        self._check()
                        axis[start:start + original._VECTOR_UNITS] = np.arange(
                            start, min(units, start + original._VECTOR_UNITS))
                groups = []
                for _, location in slots:
                    self._check()
                    group = self._h5.create_group(location)
                    group.attrs["complete"] = False
                    group.create_dataset("source_file_features", (units, 224), dtype="<f4",
                                         chunks=(min(256, units), 224))
                    for name in ("full_layer_C_index", "full_layer_Fortran_index"):
                        group.create_dataset(name, (units,), dtype="<i8")
                    group.create_dataset("valid_mask", (units, 224), dtype="?",
                                         chunks=(min(256, units), 224))
                    groups.append(group)
                moment_path = "decoders/" + owner + "/feature_moments/" + layer
                first_owner = moment_path not in self._h5
                moments = self._h5.require_group(moment_path)
                if first_owner:
                    moments.attrs["complete"] = False
                    for name in ("mean", "std"):
                        moments.create_dataset(name, (units,), dtype="<f4")
                else:
                    original._require(bool(moments.attrs["complete"]),
                                      "complete_frozen_owner_moments_required")
                cursor = 0
                iterator = begin()
                for decoder in iterator:
                    self._check()
                    original._require(isinstance(decoder, prediction.DecoderBlock)
                        and decoder.participant == owner and decoder.layer == layer,
                        "exact_owner_layer_decoder_block_required")
                    count, f_ids, c_ids = prediction._decoder_block(decoder, self.axes[owner], self.budget.check)
                    original._require(all(not getattr(decoder, name).flags.writeable for name in (
                        "full_layer_Fortran_index", "full_layer_C_index", "selected_native_voxel_ids",
                        "eligible_columns", "weights", "bias", "target_mean", "target_std")),
                        "immutable_shared_decoder_block_required")
                    original._require(not bool(seen[c_ids].any()),
                                      "exact_unique_prediction_F_C_coordinates_required")
                    order = np.argsort(c_ids)
                    sorted_ids = c_ids[order]
                    for name, supplied in (("mean", decoder.target_mean), ("std", decoder.target_std)):
                        if first_owner:
                            moments[name][sorted_ids] = supplied[order]
                        else:
                            original._require(np.array_equal(moments[name][sorted_ids], supplied[order]),
                                              "frozen_decoder_feature_moments_changed_between_methods")
                    stop = cursor + count
                    for (stream, value), group in zip(prepared, groups):
                        self._check()
                        predicted = prediction.predict_decoder_block(value, decoder,
                            check=self._check, row_chunk_size=64)
                        original._require(isinstance(predicted, prediction.FeaturePredictionBlock)
                            and (predicted.method, predicted.source, predicted.target) == stream
                            and predicted.decoder_participant == owner and predicted.layer == layer
                            and predicted.channel == decoder.channel and predicted.image_ids == self.image_ids
                            and np.array_equal(predicted.full_layer_C_index, c_ids)
                            and np.array_equal(predicted.full_layer_Fortran_index, f_ids),
                            "prediction_stream_owner_image_and_F_C_binding_required")
                        values = predicted.source_file_features
                        metrics._float32(values, (224, count), "source_file_float32_prediction_rows_required")
                        original._require(bool(np.isfinite(values).all()), "nonfinite_decoded_source_file_features")
                        group["source_file_features"][cursor:stop] = values.T
                        group["valid_mask"][cursor:stop] = np.ones((count, 224), dtype=bool)
                        group["full_layer_C_index"][cursor:stop] = c_ids
                        group["full_layer_Fortran_index"][cursor:stop] = f_ids
                        del predicted, values
                    cursor = stop
                    seen[c_ids] = True
                original._require(cursor == units and bool(seen.all()),
                                  "complete_all_channel_decoded_layer_required")
                # A close failure is an operation failure, before any shared
                # moments or stream is marked complete. Close only once.
                finish()
                self._check()
                moments.attrs["complete"] = True
                for group in groups:
                    group.attrs["complete"] = True
                self._h5.flush()
                self._check()
                for key, location in slots:
                    record = self._record(key, location, "decoded")
                    record.update(stored_shape=[units, 224], storage_axes=["append_order_unit", "image"],
                        feature_moments=moment_path, coordinates={name: location + "/" + name for name in
                        ("full_layer_C_index", "full_layer_Fortran_index")},
                        values=location + "/source_file_features", valid_mask=location + "/valid_mask")
