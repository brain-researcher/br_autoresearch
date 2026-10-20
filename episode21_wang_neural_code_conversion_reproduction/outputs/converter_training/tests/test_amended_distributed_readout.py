"""Three changed-interface checks; coordinator compute only, no study assets.

Reuse the original composition and one-fc8 persistence fixtures. Numerical
inference is already qualified separately; these checks concern the amended
full roster, policy binding, phase joins and unchanged sole-writer lifecycle.
"""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import resource
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import amended_distributed_readout as pipeline
import amended_readout_publication as publication
import test_finding_a_collection as collection_fixture
import test_finding_a_distributed_readout as distributed_fixture
import test_finding_a_staged_collection as staged_fixture


def full_reader():
    reader = distributed_fixture.reader_fixture()
    for stream in pipeline.collection.STREAMS:
        method, source, target = stream
        reader.fitted_object_ids[stream] = (
            pipeline.assembly.MANIFEST + "#participant=" + source if method == "within_source" else
            str(pipeline.models._model_path(method, source, target).relative_to(pipeline.models.BASE)))
    for record in reader.records.values():
        record.update(status="decoded", reason=None)
    return reader


def amended_stage(path):
    with publication.AmendedStagedCollection(path, **collection_fixture.settings()) as writer:
        key = collection_fixture.KEY
        writer.write_layer(method=key[0], source=key[1], target=key[2], layer=key[3],
                           blocks=collection_fixture.blocks())
        collection_fixture.seal(writer)
        results = collection_fixture.completed_results(writer)
        published = writer.publish_decoded_point(results[0].point)
    return published, results


class AmendedDistributedReadoutTests(unittest.TestCase):
    def test_decode_uses_all_45_prepared_streams_and_all_855_slots_without_fallback(self):
        for failure in (None, "incomplete", "denied"):
            with self.subTest(failure=failure), ExitStack() as stack:
                reader, context = full_reader(), object()
                budget = SimpleNamespace(allocation={"job_id": "fixture", "hostname": "compute"})
                resolver = mock.Mock(return_value=None,
                    side_effect=PermissionError("not admitted") if failure == "denied" else None)
                bound = SimpleNamespace(image_ids=reader.image_ids, feature_identity=reader.feature_identity,
                    truth_reader=object(), check=mock.Mock(), close=mock.Mock())
                prepared = SimpleNamespace(model_policy=pipeline.MODEL_POLICY,
                    unavailable_reasons={}, fitted_object_ids=reader.fitted_object_ids,
                    prepared={stream: SimpleNamespace(decoder_axis=reader.axes[stream[2] or stream[1]])
                              for stream in pipeline.collection.STREAMS}, check=mock.Mock())
                if failure == "incomplete":
                    prepared.prepared.pop(pipeline.collection.STREAMS[-1])
                written, lifetimes = [], []
                point, manifest = object(), object()
                writer = SimpleNamespace(seal_decoded=mock.Mock(), result_sink=mock.Mock(),
                    block_provider=mock.Mock(return_value="block"),
                    publish_decoded_point=mock.Mock(return_value=(Path("/fixture/stage.h5"),
                        Path("/fixture/stage.decoded_point.json"))))

                def write_owner_layer(*, owner, layer, prepared_by_stream, blocks):
                    self.assertEqual(blocks, (owner, layer))
                    expected = {stream for stream in pipeline.collection.STREAMS
                                if (stream[2] or stream[1]) == owner}
                    self.assertEqual(set(prepared_by_stream), expected)
                    self.assertEqual(len(expected), 9)
                    written.extend((*stream, layer) for stream in prepared_by_stream)

                writer.write_owner_layer = write_owner_layer

                @contextmanager
                def make_writer(path, **kwargs):
                    self.assertEqual(path, Path("/fixture/stage.h5"))
                    self.assertEqual(kwargs["image_ids"], reader.image_ids)
                    self.assertEqual(dict(kwargs["fitted_object_ids"]), reader.fitted_object_ids)
                    self.assertEqual(sum("brain_constant_training/" in str(value)
                                         for value in kwargs["fitted_object_ids"].values()), 20)
                    kwargs["require_admitted"]()
                    try:
                        yield writer
                    finally:
                        lifetimes.append("writer_closed")

                def evaluate(provider, training, image_ids, *, check, result_sink):
                    self.assertIs(training, reader.training)
                    self.assertEqual(image_ids, reader.image_ids)
                    self.assertEqual(len(written), 855)
                    self.assertEqual(set(written), set(pipeline.collection.SLOTS))
                    writer.seal_decoded.assert_called_once_with()
                    self.assertEqual(provider(method="within_source", source="sub-01", target=None,
                                              layer="fc8"), "block")
                    self.assertIs(writer.block_provider.call_args.kwargs["truth_provider"], bound.truth_reader)
                    result_sink(point)
                    check()
                    return point

                stack.enter_context(mock.patch.object(pipeline.assembly, "_require_reader_context"))
                stack.enter_context(mock.patch.object(pipeline.collection, "_require_file_limit"))
                manifest_call = stack.enter_context(mock.patch.object(pipeline.assembly,
                    "load_complete_manifest", return_value=manifest))
                stack.enter_context(mock.patch.object(pipeline, "_decoder_metadata",
                    return_value=(reader.axes, reader.training)))
                stack.enter_context(mock.patch.object(pipeline, "_decoder_blocks",
                    side_effect=lambda manifest, axis, layer, budget, check: (axis.participant, layer)))
                bind = stack.enter_context(mock.patch.object(pipeline, "_bind_inputs", return_value=bound))
                prepare = stack.enter_context(mock.patch.object(pipeline, "_prepare_models", return_value=prepared))
                create = stack.enter_context(mock.patch.object(publication, "AmendedStagedCollection", side_effect=make_writer))
                evaluate_call = stack.enter_context(mock.patch.object(pipeline.inference, "evaluate_panel", side_effect=evaluate))
                draws = stack.enter_context(mock.patch.object(pipeline.batches, "evaluate_draw_batch"))
                arguments = dict(require_admitted_context=resolver, budget=budget,
                    stage_h5="/fixture/stage.h5", maximum_bytes=123456,
                    authenticated_tau_train_image_ids=("authenticated-tau",))
                if failure:
                    with self.assertRaises(PermissionError if failure == "denied" else ValueError):
                        pipeline.run_decoded_point(context, **arguments)
                    create.assert_not_called(); evaluate_call.assert_not_called()
                    if failure == "denied":
                        manifest_call.assert_not_called(); bind.assert_not_called(); prepare.assert_not_called()
                        bound.close.assert_not_called()
                    else:
                        bound.close.assert_called_once_with()
                else:
                    result = pipeline.run_decoded_point(context, **arguments)
                    self.assertEqual(result, pipeline.AmendedPhaseResult("decode", writer.publish_decoded_point.return_value))
                    self.assertEqual(result.model_policy, pipeline.MODEL_POLICY)
                    self.assertEqual(lifetimes, ["writer_closed"])
                    bound.close.assert_called_once_with()
                    writer.publish_decoded_point.assert_called_once_with(point)
                    writer.result_sink.assert_called_once_with(point)
                    self.assertEqual(prepare.call_args.kwargs["authenticated_tau_train_image_ids"],
                                     ("authenticated-tau",))
                    self.assertEqual(resolver.call_args.kwargs["operation"], pipeline.DECODE)
                    self.assertEqual(resolver.call_args.kwargs["execution_options"]["model_policy"], pipeline.MODEL_POLICY)
                draws.assert_not_called()

    def test_draw_and_final_join_preserve_canonical_batches_and_reader_closure(self):
        for phase, failure in (("draw", None), ("draw", "scope"), ("draw", "policy"),
                ("draw", "denied"), ("finalize", None), ("finalize", "workers"), ("finalize", "batch")):
            with self.subTest(phase=phase, failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root, reader, context = Path(directory).resolve(), full_reader(), object()
                budget = SimpleNamespace(allocation={"job_id": "fixture", "hostname": "compute"})
                resolver = mock.Mock(return_value=None,
                    side_effect=PermissionError("closed") if failure == "denied" else None)
                if failure == "scope":
                    reader.fitted_object_ids[("brain_loss", "sub-01", "sub-03")] = None
                if failure == "policy":
                    opener_error = ValueError("explicit_adapted_stage_model_policy_required")
                else:
                    opener_error = None
                stack.enter_context(mock.patch.object(pipeline.assembly, "_require_reader_context"))
                opener = stack.enter_context(mock.patch.object(publication, "open_decoded_point",
                    return_value=reader, side_effect=opener_error))
                if phase == "draw":
                    truth, batch = SimpleNamespace(close=mock.Mock()), object()
                    truth_factory = stack.enter_context(mock.patch.object(pipeline, "DirectTruthReader", return_value=truth))

                    def evaluate(provider, training, image_ids, *, batch_index, check):
                        self.assertEqual(batch_index, 62)
                        self.assertIs(training, reader.training)
                        self.assertEqual(image_ids, reader.image_ids)
                        reader.close.assert_not_called(); truth.close.assert_not_called()
                        check()
                        self.assertEqual(provider(method="brain_loss", source="sub-03", target="sub-01", layer="fc8"),
                                         "provided")
                        self.assertIs(reader.block_provider.call_args.kwargs["truth_provider"], truth)
                        return batch

                    def write(path, actual, **kwargs):
                        reader.close.assert_called_once_with(); truth.close.assert_called_once_with()
                        self.assertIs(actual, batch)
                        self.assertEqual(kwargs["expected_stage_identity"], reader.stage_identity)
                        self.assertEqual(kwargs["expected_image_ids"], reader.image_ids)
                        self.assertEqual(kwargs["expected_batch_index"], 62)
                        self.assertEqual(kwargs["maximum_bytes"], 12345)
                        kwargs["check"]()
                        return path

                    draw = stack.enter_context(mock.patch.object(pipeline.batches, "evaluate_draw_batch", side_effect=evaluate))
                    publish = stack.enter_context(mock.patch.object(pipeline.artifacts, "write_draw_batch", side_effect=write))
                    arguments = dict(require_admitted_context=resolver, budget=budget, stage_h5=root / "stage.h5",
                        maximum_collection_bytes=123456, batch_index=62, batch_path=root / "draw_batch_062.json",
                        maximum_batch_bytes=12345)
                    if failure:
                        with self.assertRaises(PermissionError if failure == "denied" else ValueError):
                            pipeline.run_draw_batch(context, **arguments)
                        publish.assert_not_called(); truth_factory.assert_not_called(); draw.assert_not_called()
                        if failure == "denied":
                            opener.assert_not_called()
                    else:
                        result = pipeline.run_draw_batch(context, **arguments)
                        self.assertEqual(result, pipeline.AmendedPhaseResult("draw", (root / "draw_batch_062.json",)))
                else:
                    closed = mock.Mock(return_value=None,
                        side_effect=PermissionError("workers live") if failure == "workers" else None)
                    visited, primary = [], SimpleNamespace(point=reader.point)
                    incoming, dyadic = object(), object()

                    def read(path, **kwargs):
                        reader.close.assert_called_once_with()
                        index = kwargs["expected_batch_index"]
                        self.assertEqual(path, root / "batches" / f"draw_batch_{index:03d}.json")
                        self.assertEqual(kwargs["expected_image_ids"], reader.image_ids)
                        self.assertEqual(kwargs["expected_stage_identity"], reader.stage_identity)
                        if failure == "batch" and index == 37:
                            raise FileNotFoundError("batch37")
                        visited.append(index)
                        return index

                    def assemble(point, batch_iterator, *, check):
                        self.assertIs(point, reader.point)
                        self.assertEqual(tuple(batch_iterator), tuple(range(63)))
                        check()
                        return primary

                    def finalize(stage, final, actual_primary, actual_incoming, actual_dyadic, **kwargs):
                        self.assertEqual(visited, list(range(63)))
                        self.assertIs(actual_primary, primary)
                        self.assertIs(actual_incoming, incoming)
                        self.assertIs(actual_dyadic, dyadic)
                        self.assertGreaterEqual(closed.call_count, 2)
                        kwargs["require_readers_closed"]()
                        return final, final.with_suffix(".decoded_feature_manifest.json"), final.with_suffix(".finding_a_report.json")

                    read_call = stack.enter_context(mock.patch.object(pipeline.artifacts, "read_draw_batch", side_effect=read))
                    join = stack.enter_context(mock.patch.object(pipeline.batches, "assemble_bootstrap", side_effect=assemble))
                    inc = stack.enter_context(mock.patch.object(pipeline.secondary, "target_incoming", return_value=incoming))
                    dyad = stack.enter_context(mock.patch.object(pipeline.secondary, "dyadic_bootstrap", return_value=dyadic))
                    publish = stack.enter_context(mock.patch.object(publication, "finalize_decoded_point", side_effect=finalize))
                    arguments = dict(require_admitted_context=resolver, budget=budget, stage_h5=root / "stage.h5",
                        final_h5=root / "final.h5", batch_directory=root / "batches", maximum_collection_bytes=123456,
                        maximum_batch_bytes=12345, require_readers_closed=closed)
                    if failure:
                        with self.assertRaises(PermissionError if failure == "workers" else FileNotFoundError):
                            pipeline.run_finalizer(context, **arguments)
                        publish.assert_not_called(); inc.assert_not_called(); dyad.assert_not_called()
                        if failure == "workers":
                            opener.assert_not_called(); join.assert_not_called(); read_call.assert_not_called()
                    else:
                        result = pipeline.run_finalizer(context, **arguments)
                        self.assertEqual(result.phase, "finalize")
                        self.assertEqual(len(result.artifacts), 3)
                        self.assertEqual(read_call.call_count, 63)
                        publish.assert_called_once()
                        self.assertIs(inc.call_args.args[0], reader.point)
                        self.assertIs(dyad.call_args.args[0], reader.point)
                if failure is None:
                    self.assertEqual(result.model_policy, pipeline.MODEL_POLICY)
                    self.assertEqual(resolver.call_args.kwargs["operation"],
                                     pipeline.DRAW if phase == "draw" else pipeline.FINALIZE)
                    self.assertEqual(resolver.call_args.kwargs["execution_options"]["model_policy"], pipeline.MODEL_POLICY)

    def test_policy_bound_publication_retains_inode_identity_caps_and_final_report_atomicity(self):
        collection_fixture.FindingACollectionTests.setUp(self)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(pipeline.assembly,
                "_require_reader_context", side_effect=lambda budget: budget.check()):
            root = Path(directory).resolve()
            original_path = root / "original.h5"
            staged_fixture.make_stage(original_path)
            with self.assertRaisesRegex(ValueError, "explicit_adapted_stage_model_policy_required"):
                publication.open_decoded_point(original_path, **staged_fixture.read_arguments())
            for late_failure in (False, True):
                with self.subTest(late_failure=late_failure):
                    path, final = root / f"stage{late_failure}.h5", root / f"final{late_failure}.h5"
                    (stage, metadata), results = amended_stage(path)
                    historical, inode = metadata.read_bytes(), (path.stat().st_dev, path.stat().st_ino)
                    with publication.open_decoded_point(path, **staged_fixture.read_arguments()) as reader:
                        self.assertEqual(reader._state._h5.attrs["model_policy"], pipeline.MODEL_POLICY)
                        self.assertEqual(dict(reader.fitted_object_ids), collection_fixture.settings()["fitted_object_ids"])
                    with self.assertRaisesRegex(PermissionError, "readers live"):
                        publication.finalize_decoded_point(path, final, *results,
                            **staged_fixture.read_arguments(), require_readers_closed=mock.Mock(
                                side_effect=PermissionError("readers live")))
                    self.assertFalse(path.with_suffix(".finalization_started.json").exists())
                    cap = path.stat().st_size + metadata.stat().st_size - 1
                    prior_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
                    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, prior_limit[1]))
                    try:
                        with self.assertRaisesRegex(ValueError, "bounded_stage_files_required"):
                            publication.open_decoded_point(path, **{**staged_fixture.read_arguments(),
                                "maximum_bytes": cap})
                    finally:
                        resource.setrlimit(resource.RLIMIT_FSIZE, prior_limit)
                    link = publication.original.os.link
                    report_path = final.with_suffix(".finding_a_report.json")

                    def guarded_link(source, destination, *args, **kwargs):
                        if late_failure and Path(destination) == report_path:
                            raise OSError("final report link failure")
                        return link(source, destination, *args, **kwargs)

                    with mock.patch.object(publication.original.os, "link", side_effect=guarded_link), \
                            mock.patch.object(publication.original.prediction, "predict_decoder_block",
                                side_effect=AssertionError("publication must not refit or predict")):
                        if late_failure:
                            with self.assertRaisesRegex(OSError, "final report link failure"):
                                publication.finalize_decoded_point(path, final, *results,
                                    **staged_fixture.read_arguments(), require_readers_closed=lambda: None)
                            self.assertFalse(report_path.exists())
                            self.assertTrue(Path(str(report_path) + ".partial").exists())
                        else:
                            published = publication.finalize_decoded_point(path, final, *results,
                                **staged_fixture.read_arguments(), require_readers_closed=lambda: None)
                            self.assertEqual(published[0], final)
                            report = json.loads(report_path.read_text())
                            self.assertEqual(report["model_policy"], pipeline.MODEL_POLICY)
                            self.assertEqual(report["status"], "complete_requested_record")
                            self.assertEqual(report["quantitative_finding"], "not_evaluable")
                            self.assertEqual(report["comparability"]["reason"], "missing_outcome_independent_margin")
                    self.assertEqual((final.stat().st_dev, final.stat().st_ino), inode)
                    self.assertEqual(metadata.read_bytes(), historical)
                    manifest = json.loads(final.with_suffix(".decoded_feature_manifest.json").read_text())
                    self.assertEqual(manifest["model_policy"], pipeline.MODEL_POLICY)
                    self.assertEqual((len(manifest["slots"]), manifest["decoded_slots"], manifest["unavailable_slots"]),
                                     (855, 1, 854))
                    with h5py.File(final, "r") as handle:
                        self.assertEqual(handle.attrs["model_policy"], pipeline.MODEL_POLICY)
                        self.assertTrue(handle.attrs["complete"])
                        self.assertEqual(handle["resampling/image_draw_indices"].shape, (1000, 224))
                        self.assertEqual(handle["resampling/dyadic_draw_indices"].shape[0], 1000)
                    with self.assertRaisesRegex(ValueError, "finalization_already_claimed"):
                        publication.open_decoded_point(path, **staged_fixture.read_arguments())


if __name__ == "__main__":
    unittest.main()
