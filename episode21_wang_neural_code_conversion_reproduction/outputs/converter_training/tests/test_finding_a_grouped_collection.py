"""Three grouped-write checks, unrun until coordinated Slurm execution.

The existing full-fc8 collection fixture supplies tiny native axes and all224
rows. Existing prediction/collection tests own numerical and fatal-cap coverage;
these tests compare grouping to the unchanged per-stream writer and exercise
shared incomplete state. No models, empirical inputs or access grants are used.
"""
from dataclasses import replace
import json
from pathlib import Path
import resource
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_grouped_collection as grouped
import finding_a_collection as original
import test_finding_a_collection as collection_fixtures
from test_finding_a_collection import (IMAGES, KEY, settings, blocks as old_blocks,
    readonly, truth_provider, completed_results)

prediction = original.prediction
OWNER, LAYER = KEY[2], KEY[3]
ARRAY_FIELDS = ("full_layer_Fortran_index", "full_layer_C_index", "selected_native_voxel_ids",
                "eligible_columns", "weights", "bias", "target_mean", "target_std")


def fixture():
    arguments = settings()
    arguments["fitted_object_ids"][("within_source", OWNER, None)] = "fixture-within-decoder"
    streams = tuple(stream for stream in original.STREAMS
        if original.metrics._owner(*stream) == OWNER and arguments["fitted_object_ids"][stream] is not None)
    prepared = {}
    for number, stream in enumerate(streams):
        rows = np.arange(224, dtype=np.float32)[:, None] / np.float32(32)
        values = (rows * np.float32(number + 1) + np.array([1, -2, .5, 3], dtype=np.float32)[None])
        prepared[stream] = prediction.PreparedDecoderInput(*stream,
            arguments["decoder_axes_by_owner"][OWNER], IMAGES, readonly(values))
    decoders = []
    for _, decoder in old_blocks():
        ids = decoder.full_layer_C_index
        decoder = replace(decoder,
            weights=(np.tile(np.array([.5, -.25, .125, .0625], dtype=np.float32), (len(ids), 1))
                     * (np.float32(1) + (ids % 7).astype(np.float32)[:, None] / np.float32(8))),
            bias=ids.astype(np.float32) / np.float32(64) - np.float32(.25))
        for name in ARRAY_FIELDS:
            getattr(decoder, name).setflags(write=False)
        decoders.append(decoder)
    # Valid append order differs from C order while every individual decoder
    # block remains contiguous in its declared native F coordinates.
    return arguments, prepared, tuple(decoders[index] for index in (2, 0, 3, 1))


def seal(writer, streams):
    decoded = {(*stream, LAYER) for stream in streams}
    for key in original.SLOTS:
        if key not in decoded:
            writer.unavailable_layer(method=key[0], source=key[1], target=key[2],
                                     layer=key[3], reason="fixture_model_not_available")
    writer.seal_decoded()


class FindingAGroupedCollectionTests(unittest.TestCase):
    def setUp(self):
        # The caller owns/restores the inherited finite cap and default signal;
        # the production subclass does not set either policy.
        collection_fixtures.FindingACollectionTests.setUp(self)

    def test_grouped_prediction_replay_and_publication_equal_original_writer(self):
        arguments, prepared, decoders = fixture()
        self.assertEqual(len(prepared), 3)
        policies = resource.getrlimit(resource.RLIMIT_FSIZE), signal.getsignal(signal.SIGXFSZ)
        self.assertIs(grouped.GroupedFindingACollection.__init__, original.FindingACollection.__init__)
        for method in ("block_provider", "result_sink", "seal_decoded", "publish", "close"):
            self.assertIs(getattr(grouped.GroupedFindingACollection, method),
                          getattr(original.FindingACollection, method))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(original.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            first, second = Path(directory) / "original.h5", Path(directory) / "grouped.h5"
            with original.FindingACollection(first, **arguments) as writer:
                for stream, value in prepared.items():
                    predicted = ((prediction.predict_decoder_block(value, decoder,
                        check=writer._check, row_chunk_size=64), decoder) for decoder in decoders)
                    writer.write_layer(method=stream[0], source=stream[1], target=stream[2],
                                       layer=LAYER, blocks=predicted)
                seal(writer, prepared)
                writer.publish(*completed_results(writer))

            visited, closed, calls = [], [], []
            predictor = prediction.predict_decoder_block

            def source():
                try:
                    for decoder in decoders:
                        visited.append(id(decoder))
                        yield decoder
                finally:
                    closed.append(True)

            def predict(value, decoder, *, check, row_chunk_size):
                calls.append(((value.method, value.source, value.target), id(decoder), row_chunk_size))
                return predictor(value, decoder, check=check, row_chunk_size=row_chunk_size)

            with grouped.GroupedFindingACollection(second, **arguments) as writer, \
                    mock.patch.object(prediction, "predict_decoder_block", side_effect=predict):
                # Caller mapping insertion order must not change stream order.
                writer.write_owner_layer(owner=OWNER, layer=LAYER,
                    prepared_by_stream=dict(reversed(tuple(prepared.items()))), blocks=source())
                self.assertEqual(visited, [id(value) for value in decoders])
                self.assertEqual(closed, [True])
                self.assertEqual(calls, [(stream, id(decoder), 64) for decoder in decoders for stream in prepared])
                self.assertEqual(len(writer._records), 3)
                self.assertTrue(writer._h5["decoders/" + OWNER + "/feature_moments/" + LAYER].attrs["complete"])
                seal(writer, prepared)
                for stream, value in prepared.items():
                    replay = list(writer.block_provider(method=stream[0], source=stream[1], target=stream[2],
                                                       layer=LAYER, truth_provider=truth_provider))
                    expected = np.concatenate([predictor(value, decoder, check=lambda: None,
                        row_chunk_size=64).source_file_features for decoder in decoders], axis=1)
                    np.testing.assert_array_equal(np.concatenate([block.predicted for block in replay], axis=1), expected)
                    np.testing.assert_array_equal(np.concatenate([block.full_layer_C_index for block in replay]),
                                                  np.concatenate([block.full_layer_C_index for block in decoders]))
                writer.publish(*completed_results(writer))

            with h5py.File(first, "r") as expected, h5py.File(second, "r") as actual:
                names = []
                expected.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
                actual_names = []
                actual.visititems(lambda name, item: actual_names.append(name) if isinstance(item, h5py.Dataset) else None)
                self.assertEqual(names, actual_names)
                for name in names:
                    self.assertEqual(actual[name].dtype, expected[name].dtype)
                    np.testing.assert_array_equal(actual[name][...], expected[name][...])
                self.assertTrue(actual.attrs["complete"])
            manifests = [json.loads(path.with_suffix(".decoded_feature_manifest.json").read_text())
                         for path in (first, second)]
            for manifest in manifests:
                manifest.pop("hdf5")
                self.assertEqual((manifest["decoded_slots"], manifest["unavailable_slots"]), (3, 852))
            self.assertEqual(*manifests)
            report = json.loads(second.with_suffix(".finding_a_report.json").read_text())
            self.assertEqual(report["quantitative_finding"], "not_evaluable")
            self.assertEqual(report["comparability"]["status"], "not_evaluable")
        self.assertEqual(policies, (resource.getrlimit(resource.RLIMIT_FSIZE), signal.getsignal(signal.SIGXFSZ)))

    def test_complete_roster_native_coordinates_and_shared_moments_are_bound(self):
        arguments, prepared, decoders = fixture()
        streams = tuple(prepared)

        class Unopened:
            def __init__(self):
                self.opened = self.closed = False

            def __iter__(self):
                self.opened = True
                raise AssertionError("decoder opened before full stream admission")

            def close(self):
                self.closed = True

        bad_owner = replace(prepared[streams[-1]], decoder_axis=arguments["decoder_axes_by_owner"][original.PARTICIPANTS[2]])
        bad_image = replace(prepared[streams[-1]], image_ids=IMAGES[::-1])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(original.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            for number, (mapping, reason) in enumerate((
                    ({key: value for key, value in prepared.items() if key != streams[-1]}, "all_present"),
                    ({**prepared, streams[-1]: bad_owner}, "decoder_axis"),
                    ({**prepared, streams[-1]: bad_image}, "identity"))):
                source = Unopened()
                with grouped.GroupedFindingACollection(Path(directory) / ("identity%d.h5" % number), **arguments) as writer:
                    with self.assertRaisesRegex(ValueError, reason):
                        writer.write_owner_layer(owner=OWNER, layer=LAYER, prepared_by_stream=mapping, blocks=source)
                    self.assertTrue(writer._failed and writer._closed)
                    self.assertFalse(writer._records)
                self.assertFalse(source.opened)
                self.assertTrue(source.closed)

            bad_coordinates = replace(decoders[0], full_layer_C_index=readonly(decoders[0].full_layer_C_index + 1))
            mutable = replace(decoders[0], weights=decoders[0].weights.copy())
            for name, supplied, reason in (("missing", decoders[:-1], "complete_all_channel"),
                    ("duplicate", (decoders[0], decoders[0]), "exact_unique"),
                    ("coordinates", (bad_coordinates,), "coordinates"),
                    ("mutable", (mutable,), "immutable_shared")):
                path = Path(directory) / (name + ".h5")
                with grouped.GroupedFindingACollection(path, **arguments) as writer:
                    with self.assertRaisesRegex(ValueError, reason):
                        writer.write_owner_layer(owner=OWNER, layer=LAYER, prepared_by_stream=prepared, blocks=iter(supplied))
                    self.assertFalse(writer._records)
                    self.assertTrue(writer._closed)
                with h5py.File(str(path) + ".partial", "r") as partial:
                    self.assertFalse(partial["decoders/" + OWNER + "/feature_moments/" + LAYER].attrs["complete"])
                    self.assertTrue(all(not group.attrs["complete"] for group in partial["layers"].values()))
                self.assertFalse(path.exists())
            for complete in (False, True):
                path = Path(directory) / ("moments%d.h5" % complete)
                with grouped.GroupedFindingACollection(path, **arguments) as writer:
                    # A supplied incomplete/mismatched shared group must never
                    # be promoted merely because several streams use it.
                    moments = writer._h5.require_group("decoders/" + OWNER + "/feature_moments/" + LAYER)
                    moments.attrs["complete"] = complete
                    for name in ("mean", "std"):
                        moments.create_dataset(name, data=np.ones(1000, dtype=np.float32))
                    with self.assertRaisesRegex(ValueError, "frozen_owner_moments|moments_changed"):
                        writer.write_owner_layer(owner=OWNER, layer=LAYER, prepared_by_stream=prepared, blocks=iter(decoders))
                    self.assertFalse(writer._records)

    def test_cancellation_and_upstream_close_failure_leave_every_shared_slot_incomplete(self):
        arguments, prepared, decoders = fixture()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(original.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            cancelled, closed = [], []

            def check():
                if cancelled:
                    raise TimeoutError("grouped fixture deadline")

            def interrupted():
                try:
                    yield decoders[0]
                    cancelled.append(True)
                    yield decoders[1]
                finally:
                    closed.append(True)
                    raise RuntimeError("fixture secondary decoder cleanup")

            path = Path(directory) / "cancelled.h5"
            with grouped.GroupedFindingACollection(path, **{**arguments, "budget": SimpleNamespace(check=check)}) as writer:
                with self.assertRaisesRegex(TimeoutError, "fixture deadline") as failure:
                    writer.write_owner_layer(owner=OWNER, layer=LAYER, prepared_by_stream=prepared, blocks=interrupted())
                self.assertEqual(failure.exception.__notes__, ["grouped decoder cleanup failed: RuntimeError"])
                self.assertTrue(writer._closed and writer._failed and not writer._h5.id.valid)
                self.assertFalse(writer._records)
            self.assertEqual(closed, [True])

            class FailingClose:
                def __init__(self):
                    self.iterator = iter(decoders)
                    self.closes = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    return next(self.iterator)

                def close(self):
                    self.closes += 1
                    raise OSError("fixture source close failed")

            failed_source = FailingClose()
            path2 = Path(directory) / "sourceclose.h5"
            with grouped.GroupedFindingACollection(path2, **arguments) as writer:
                with self.assertRaisesRegex(OSError, "source close failed"):
                    writer.write_owner_layer(owner=OWNER, layer=LAYER, prepared_by_stream=prepared, blocks=failed_source)
                self.assertTrue(writer._closed and writer._failed)
                self.assertFalse(writer._records)
            self.assertEqual(failed_source.closes, 1)
            for destination in (path, path2):
                with h5py.File(str(destination) + ".partial", "r") as partial:
                    self.assertFalse(partial.attrs["complete"])
                    self.assertFalse(partial["decoders/" + OWNER + "/feature_moments/" + LAYER].attrs["complete"])
                    self.assertTrue(all(not group.attrs["complete"] for group in partial["layers"].values()))
                for final in (destination, destination.with_suffix(".decoded_feature_manifest.json"),
                              destination.with_suffix(".finding_a_report.json")):
                    self.assertFalse(final.exists())
                self.assertLessEqual(Path(str(destination) + ".partial").stat().st_size, arguments["maximum_bytes"])


if __name__ == "__main__":
    unittest.main()
