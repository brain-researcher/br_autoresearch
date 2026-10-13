"""Dormant readout composition checks; no empirical data or numerical campaign."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_first_readout as readout


IDS = tuple(f"image-{index:03d}" for index in range(224))


def model_metadata():
    # The existing artifact loader has its own numerical/state tests. This
    # fixture supplies only the additional scope fields inspected by this caller.
    return {"initialization": {"pair_index": 4, "source": "sub-03", "target": "sub-01",
                               "input_width": 12897, "output_width": 12294},
            "completion": {"backend": "sparse"}}


class FindingAFirstReadoutTests(unittest.TestCase):
    def test_denied_or_incompatible_original_model_precedes_tau_and_publication(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory).resolve() / "collection.h5"
            context = object()
            budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"})
            stack.enter_context(mock.patch.object(readout.assembly, "_require_reader_context"))
            cap = stack.enter_context(mock.patch.object(readout.collection, "_require_file_limit"))
            brain = stack.enter_context(mock.patch.object(readout, "_brain_disposition", return_value=readout.BRAIN_REASON))
            load = stack.enter_context(mock.patch.object(readout.artifacts, "load_final_content_artifact"))
            manifest = stack.enter_context(mock.patch.object(readout.assembly, "load_complete_manifest"))
            binding = stack.enter_context(mock.patch.object(readout.inputs, "bind_finding_a_inputs"))
            writer = stack.enter_context(mock.patch.object(readout.collection, "FindingACollection"))
            # Resolve only a source-free placeholder, never the real model path.
            stack.enter_context(mock.patch.object(readout, "FINAL_MODEL", path.parent / "original.pt"))

            def run(resolver):
                return readout.run_finding_a_first_readout(context, require_admitted_context=resolver,
                    budget=budget, collection_path=path, maximum_bytes=1234567)

            with self.assertRaises(PermissionError):
                run(mock.Mock(side_effect=PermissionError("closed")))
            for item in (cap, brain, load, manifest, binding, writer):
                item.assert_not_called()
            cap.side_effect = ValueError("caller fatal cap absent")
            with self.assertRaises(ValueError):
                run(lambda *args, **kwargs: None)
            for item in (brain, load, manifest, binding, writer):
                item.assert_not_called()
            cap.side_effect = None

            for section, key, value in (("initialization", "pair_index", 5),
                    ("initialization", "source", "sub-05"), ("initialization", "target", "sub-05"),
                    ("initialization", "input_width", 12896), ("initialization", "output_width", 12295),
                    ("completion", "backend", "dense")):
                with self.subTest(field=key):
                    artifact = model_metadata()
                    artifact[section][key] = value
                    load.return_value = artifact
                    with self.assertRaisesRegex(ValueError, "original_complete_sparse_content_pair4"):
                        run(lambda *args, **kwargs: None)
                    manifest.assert_not_called(); binding.assert_not_called(); writer.assert_not_called()
            load.side_effect = ValueError("existing loader rejects incomplete or nonoriginal artifact")
            with self.assertRaises(ValueError):
                run(lambda *args, **kwargs: None)
            manifest.assert_not_called(); binding.assert_not_called(); writer.assert_not_called()
            load.side_effect = None
            load.return_value = model_metadata()
            seen = []

            def admitted(actual, **kwargs):
                self.assertIs(actual, context)
                self.assertEqual(kwargs["operation"], "run_finding_a_first_readout")
                self.assertEqual((kwargs["source"], kwargs["target"]), ("sub-03", "sub-01"))
                self.assertIs(kwargs["budget"], budget)
                self.assertEqual(dict(kwargs["execution_options"]), {"collection_path": path, "maximum_bytes": 1234567})
                with self.assertRaises(TypeError):
                    kwargs["execution_options"]["maximum_bytes"] = 0
                seen.append(kwargs["execution_options"])

            def moved_model(*args, **kwargs):
                budget.allocation["job_id"] = "different-job"
                return model_metadata()

            load.side_effect = moved_model
            with self.assertRaisesRegex(ValueError, "cannot_move_allocations"):
                run(admitted)
            self.assertTrue(seen)
            binding.assert_not_called(); writer.assert_not_called()

        # The new preparation join rejects changed source/target training IDs
        # before either existing predictor can normalize or run a model.
        training = {owner: (f"training-{owner}",) for owner in readout.schedule.PARTICIPANTS}
        source = SimpleNamespace(participant="sub-03", role="tau_test", native_voxel_ids=range(12897),
                                 values=SimpleNamespace(shape=(224, 12897)), image_ids=IDS)
        bound = SimpleNamespace(source_responses=source, image_ids=IDS)
        for key, replacement in (("source_training_image_ids", ("different-source",)),
                                 ("target_decoder_training_image_ids", ("different-target",))):
            artifact = {"source_training_image_ids": training["sub-03"],
                        "target_decoder_training_image_ids": training["sub-01"]}
            artifact[key] = replacement
            with mock.patch.object(readout.prediction, "prepare_within_source") as within, \
                    mock.patch.object(readout.prediction, "prepare_content") as converted, \
                    self.assertRaisesRegex(ValueError, "training_identities"):
                readout._prepare_predictions(bound, artifact, {}, training, None, lambda: None)
            within.assert_not_called(); converted.assert_not_called()

    def test_complete_roster_original_inference_composition_and_cleanup(self):
        self.assertEqual(readout.inference.BOOTSTRAP_DRAWS, 1000)
        self.assertEqual(readout.secondary.DYADIC_DRAWS, 1000)
        self.assertEqual(len(readout.collection.SLOTS), 855)
        case = self
        for fail in (False, True):
            with self.subTest(cancel=fail), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                path = Path(directory).resolve() / "collection.h5"
                context = object()
                budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"})
                axes = {owner: SimpleNamespace(participant=owner) for owner in readout.schedule.PARTICIPANTS}
                training = {owner: (f"training-{owner}",) for owner in axes}
                truth = object()
                bound = SimpleNamespace(image_ids=IDS, feature_identity={"fixture": "same-truth"}, truth_reader=truth,
                                        check=mock.Mock(), close=mock.Mock())
                prepared = {readout.SELECTED_STREAMS[0]: SimpleNamespace(decoder_axis=axes["sub-03"]),
                            readout.SELECTED_STREAMS[1]: SimpleNamespace(decoder_axis=axes["sub-01"])}
                primary = SimpleNamespace(point=object())
                incoming, dyadic, point_score = object(), object(), object()
                published = (path, path.with_suffix(".manifest.json"), path.with_suffix(".report.json"))
                writers, block_calls = [], []

                class Writer:
                    def __init__(self, actual_path, **kwargs):
                        case.assertEqual(actual_path, path)
                        case.assertEqual(kwargs["image_ids"], IDS)
                        case.assertIs(kwargs["decoder_axes_by_owner"], axes)
                        case.assertIs(kwargs["decoder_training_image_ids_by_owner"], training)
                        case.assertEqual(set(kwargs["fitted_object_ids"]), set(readout.collection.STREAMS))
                        case.assertEqual({key for key, value in kwargs["fitted_object_ids"].items() if value is not None},
                                         set(readout.SELECTED_STREAMS))
                        kwargs["require_admitted"]()
                        self.written, self.missing, self.replayed, self.scores = {}, {}, [], []
                        self.sealed = self.published = False
                        self.exit_type = None
                        writers.append(self)

                    def __enter__(self):
                        return self

                    def __exit__(self, kind, *unused):
                        self.exit_type = kind

                    def write_layer(self, *, method, source, target, layer, blocks):
                        key = method, source, target, layer
                        case.assertNotIn(key, self.written)
                        self.written[key] = tuple(blocks)

                    def unavailable_layer(self, *, method, source, target, layer, reason):
                        key = method, source, target, layer
                        case.assertNotIn(key, self.missing)
                        self.missing[key] = reason

                    def seal_decoded(self):
                        case.assertEqual(set(self.written) | set(self.missing), set(readout.collection.SLOTS))
                        case.assertFalse(set(self.written) & set(self.missing))
                        case.assertEqual((len(self.written), len(self.missing)), (38, 817))
                        self.sealed = True

                    def block_provider(self, *, method, source, target, layer, truth_provider):
                        case.assertTrue(self.sealed)
                        case.assertIs(truth_provider, truth)
                        key = method, source, target, layer
                        self.replayed.append(key)
                        return self.written[key] if key in self.written else readout.inference.UnavailableLayer(self.missing[key])

                    def result_sink(self, result):
                        self.scores.append(result)

                    def publish(self, actual_primary, actual_incoming, actual_dyadic):
                        case.assertIs(actual_primary, primary)
                        case.assertIs(actual_incoming, incoming)
                        case.assertIs(actual_dyadic, dyadic)
                        self.published = True
                        return published

                def blocks(manifest, actual_prepared, axis, layer, actual_budget, check):
                    self.assertIs(actual_budget, budget)
                    self.assertIs(axis, actual_prepared.decoder_axis)
                    check()
                    block_calls.append((axis.participant, layer))
                    yield (axis.participant, layer)

                def bootstrap(provider, actual_training, actual_ids, *, check, result_sink):
                    self.assertIs(actual_training, training)
                    self.assertEqual(actual_ids, IDS)
                    self.assertTrue(writers[0].sealed)
                    check()
                    if fail:
                        raise TimeoutError("fixture inference cancellation")
                    for method, source, target in (*readout.SELECTED_STREAMS, ("brain_loss", "sub-03", "sub-01")):
                        value = provider(method=method, source=source, target=target, layer=readout.collection.LAYERS[0])
                        if method == "brain_loss":
                            self.assertIsInstance(value, readout.inference.UnavailableLayer)
                            self.assertEqual(value.reason, readout.BRAIN_REASON)
                        else:
                            self.assertEqual(len(value), 1)
                    result_sink(point_score)
                    return primary

                stack.enter_context(mock.patch.object(readout.assembly, "_require_reader_context"))
                stack.enter_context(mock.patch.object(readout.collection, "_require_file_limit"))
                stack.enter_context(mock.patch.object(readout, "_brain_disposition", return_value=readout.BRAIN_REASON))
                stack.enter_context(mock.patch.object(readout, "_load_original_model", return_value=model_metadata()))
                stack.enter_context(mock.patch.object(readout.assembly, "load_complete_manifest", return_value=object()))
                stack.enter_context(mock.patch.object(readout, "_decoder_metadata", return_value=(axes, training)))
                binder = stack.enter_context(mock.patch.object(readout.inputs, "bind_finding_a_inputs", return_value=bound))
                stack.enter_context(mock.patch.object(readout, "_prepare_predictions", return_value=prepared))
                stack.enter_context(mock.patch.object(readout, "_prediction_blocks", side_effect=blocks))
                stack.enter_context(mock.patch.object(readout.collection, "FindingACollection", Writer))
                boot = stack.enter_context(mock.patch.object(readout.batched, "bootstrap_panel", side_effect=bootstrap))
                incoming_call = stack.enter_context(mock.patch.object(readout.secondary, "target_incoming", return_value=incoming))
                dyadic_call = stack.enter_context(mock.patch.object(readout.secondary, "dyadic_bootstrap", return_value=dyadic))
                resolver = mock.Mock(return_value=None)
                kwargs = dict(require_admitted_context=resolver, budget=budget, collection_path=path, maximum_bytes=1234567)
                if fail:
                    with self.assertRaises(TimeoutError):
                        readout.run_finding_a_first_readout(context, **kwargs)
                    self.assertIs(writers[0].exit_type, TimeoutError)
                    self.assertFalse(writers[0].published)
                    incoming_call.assert_not_called(); dyadic_call.assert_not_called()
                else:
                    self.assertEqual(readout.run_finding_a_first_readout(context, **kwargs), published)
                    self.assertIsNone(writers[0].exit_type)
                    self.assertEqual(writers[0].scores, [point_score])
                    self.assertIs(incoming_call.call_args.args[0], primary.point)
                    self.assertIs(dyadic_call.call_args.args[0], primary.point)
                    self.assertEqual(set(boot.call_args.kwargs), {"check", "result_sink"})
                    self.assertEqual(set(dyadic_call.call_args.kwargs), {"check"})
                binder.assert_called_once_with(context, require_admitted_context=resolver, budget=budget)
                bound.close.assert_called_once_with()
                self.assertEqual(Counter(owner for owner, _ in block_calls), {"sub-03": 19, "sub-01": 19})
                for owner in ("sub-03", "sub-01"):
                    self.assertEqual({layer for participant, layer in block_calls if participant == owner}, set(readout.collection.LAYERS))
                reasons = list(writers[0].missing.values())
                self.assertEqual(sum(value == readout.BRAIN_REASON for value in reasons), 19)
                self.assertEqual(sum("source_inferred_not_executed" in value for value in reasons), 133)
                self.assertEqual(sum(value == readout.NOT_SELECTED for value in reasons), 665)
                for (method, source, target, _), reason in writers[0].missing.items():
                    if "source_inferred_not_executed" in reason:
                        self.assertEqual(method, "brain_loss")
                        self.assertIn("sub-01", (source, target))
                        role = "source" if source == "sub-01" else "target"
                        self.assertIn(f"sub01_{role}_fold0", reason)
                        self.assertIn(f"pair{readout.schedule.PAIR_ORDER.index((source, target))}_", reason)

        # Exercise the actual new full-layer iterator join and real F/C mapper,
        # while leaving the already-tested numerical projection mocked.
        layer = readout.collection.LAYERS[0]
        shape = readout.assembly.coverage.prefix.SHAPES[0]
        channel, spatial = 3, np.array([5, 6], dtype=np.int64)
        axis = readout.interface.NativeAxis("sub-01", np.array([2, 7]), np.array([2, 7]),
                                           np.array([0, 1]), np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32))
        raw = SimpleNamespace(participant="sub-01", layer=layer,
            full_layer_Fortran_index=channel + shape[0] * spatial,
            full_layer_C_index=channel * shape[1] * shape[2] + (spatial % shape[1]) * shape[2] + spatial // shape[1],
            selected_native_voxel_ids=np.array([[2], [7]]), native_columns=np.array([[0], [1]]),
            weights=np.ones((2, 1), dtype=np.float32), bias=np.zeros(2, dtype=np.float32),
            target_mean=np.zeros(2, dtype=np.float32), target_std=np.ones(2, dtype=np.float32))
        closed = []

        def raw_rows():
            try:
                yield raw
            finally:
                closed.append(True)

        prepared = SimpleNamespace(decoder_axis=axis)
        with mock.patch.object(readout.assembly, "iter_sparse_blocks", return_value=raw_rows()) as upstream, \
                mock.patch.object(readout.prediction, "predict_decoder_block", return_value="predicted") as project:
            iterator = readout._prediction_blocks("manifest", prepared, axis, layer, "budget", lambda: None)
            value, decoder = next(iterator)
            self.assertEqual(value, "predicted")
            self.assertEqual(decoder.channel, channel)
            np.testing.assert_array_equal(decoder.full_layer_Fortran_index, raw.full_layer_Fortran_index)
            np.testing.assert_array_equal(decoder.full_layer_C_index, raw.full_layer_C_index)
            self.assertEqual(project.call_args.kwargs["row_chunk_size"], 64)
            upstream.assert_called_once_with("manifest", participant="sub-01", layer=layer, budget="budget", block_units=256)
            iterator.close()
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
