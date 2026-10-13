"""Three source-free distributed composition checks; coordinator compute only.

Existing point, batch, collection and numerical tests own their implementations.
These fixtures verify stage joins, original scope and reader/write sequencing.
"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_distributed_readout as pipeline


IDS = tuple(f"image-{index:03d}" for index in range(224))


def reader_fixture():
    fitted = {}
    for stream in pipeline.collection.STREAMS:
        method, source, target = stream
        if method == "brain_loss" and "sub-01" in (source, target):
            fitted[stream] = None
        elif method == "within_source":
            fitted[stream] = pipeline.assembly.MANIFEST + "#participant=" + source
        else:
            fitted[stream] = str(pipeline.models._model_path(method, source, target).relative_to(pipeline.models.BASE))
    records = {}
    for key in pipeline.collection.SLOTS:
        method, source, target, layer = key
        missing = fitted[key[:3]] is None
        records[key] = dict(method=method, source=source, target=target, layer=layer,
            status="unavailable" if missing else "decoded",
            reason=pipeline.first._unavailable_reason(key[:3], pipeline.first.BRAIN_REASON) if missing else None)
    axes = {owner: SimpleNamespace(participant=owner,
        full_native_ids=range(pipeline.models.NATIVE_VOXELS[owner]),
        eligible_native_ids=range(pipeline.models.ELIGIBLE_VOXELS[owner]))
        for owner in pipeline.collection.PARTICIPANTS}
    return SimpleNamespace(records=records, fitted_object_ids=fitted, axes=axes,
        image_ids=IDS, training={owner: (owner,) for owner in axes},
        stage_identity={"fixture": "exact-stage"}, feature_identity={"fixture": "truth"},
        point=object(), close=mock.Mock(), block_provider=mock.Mock(return_value="provided"))


class DistributedFindingAReadoutTests(unittest.TestCase):
    def test_decoded_point_uses_original_point_sink_and_publishes_inference_pending(self):
        writer = SimpleNamespace(block_provider=mock.Mock(return_value="block"), result_sink=mock.Mock(),
            publish_decoded_point=mock.Mock(return_value=("stage.h5", "stage.decoded_point.json")))
        bound = SimpleNamespace(image_ids=IDS, truth_reader=object())
        training, check, point = object(), mock.Mock(), object()
        closed, calls = [], []

        @contextmanager
        def decoded(context, **kwargs):
            calls.append((context, kwargs))
            try:
                yield writer, bound, training, check
            finally:
                closed.append(True)

        def evaluate(provider, actual_training, image_ids, *, check, result_sink):
            self.assertIs(actual_training, training)
            self.assertEqual(image_ids, IDS)
            self.assertEqual(provider(method="within_source", source="sub-01", target=None, layer="fc8"), "block")
            result_sink(point)
            check()
            return point

        context, budget, resolver = object(), object(), object()
        with mock.patch.object(pipeline.full, "_decoded_full_collection", side_effect=decoded), \
                mock.patch.object(pipeline.inference, "evaluate_panel", side_effect=evaluate) as evaluate_call, \
                mock.patch.object(pipeline.batches, "evaluate_draw_batch") as draws:
            result = pipeline.run_decoded_point(context, require_admitted_context=resolver, budget=budget,
                stage_h5="/fixture/stage.h5", maximum_bytes=123456, authenticated_tau_train_image_ids=("tau",))
        self.assertEqual(result, ("stage.h5", "stage.decoded_point.json"))
        self.assertEqual(closed, [True])
        self.assertEqual(calls[0][1]["operation"], pipeline.DECODE)
        self.assertIs(calls[0][1]["collection_class"], pipeline.staged.StagedFindingACollection)
        self.assertIs(calls[0][1]["require_admitted_context"], resolver)
        self.assertEqual(calls[0][1]["authenticated_tau_train_image_ids"], ("tau",))
        writer.publish_decoded_point.assert_called_once_with(point)
        writer.result_sink.assert_called_once_with(point)
        evaluate_call.assert_called_once()
        draws.assert_not_called()

    def test_one_canonical_draw_closes_both_readers_before_publication_and_refuses_scope_drift(self):
        for failure in (None, "scope", "draw", "close", "denied"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                context = object()
                budget = SimpleNamespace(allocation={"job_id": "fixture", "hostname": "compute"})
                reader, truth, batch = reader_fixture(), SimpleNamespace(close=mock.Mock()), object()
                if failure == "scope":
                    reader.records[next(iter(reader.records))]["status"] = "unavailable"
                if failure == "close":
                    reader.close.side_effect = OSError("close failed")
                resolver = mock.Mock(return_value=None, side_effect=PermissionError("closed") if failure == "denied" else None)
                stack.enter_context(mock.patch.object(pipeline.assembly, "_require_reader_context"))
                opener = stack.enter_context(mock.patch.object(pipeline.staged, "open_decoded_point", return_value=reader))
                truth_factory = stack.enter_context(mock.patch.object(pipeline.inputs, "TruthReader", return_value=truth))

                def evaluate(provider, training, image_ids, *, batch_index, check):
                    self.assertEqual(batch_index, 62)
                    self.assertIs(training, reader.training)
                    self.assertEqual(image_ids, IDS)
                    reader.close.assert_not_called(); truth.close.assert_not_called()
                    check()
                    if failure == "draw":
                        raise TimeoutError("draw deadline")
                    self.assertEqual(provider(method="content_loss", source="sub-03", target="sub-05", layer="fc8"), "provided")
                    self.assertIs(reader.block_provider.call_args.kwargs["truth_provider"], truth)
                    return batch

                def write(path, actual, **kwargs):
                    reader.close.assert_called_once_with(); truth.close.assert_called_once_with()
                    self.assertIs(actual, batch)
                    self.assertEqual(kwargs["expected_stage_identity"], reader.stage_identity)
                    self.assertEqual(kwargs["expected_image_ids"], IDS)
                    self.assertEqual(kwargs["expected_batch_index"], 62)
                    self.assertEqual(kwargs["maximum_bytes"], 12345)
                    kwargs["check"]()
                    return path

                draw = stack.enter_context(mock.patch.object(pipeline.batches, "evaluate_draw_batch", side_effect=evaluate))
                publish = stack.enter_context(mock.patch.object(pipeline.artifacts, "write_draw_batch", side_effect=write))
                kwargs = dict(require_admitted_context=resolver, budget=budget, stage_h5=root / "stage.h5",
                    maximum_collection_bytes=123456, batch_index=62, batch_path=root / "draw_batch_062.json",
                    maximum_batch_bytes=12345)
                if failure:
                    error = {"scope": ValueError, "draw": TimeoutError, "close": OSError, "denied": PermissionError}[failure]
                    with self.assertRaises(error):
                        pipeline.run_draw_batch(context, **kwargs)
                    publish.assert_not_called()
                    if failure in ("scope", "denied"):
                        truth_factory.assert_not_called(); draw.assert_not_called()
                    if failure == "denied":
                        opener.assert_not_called(); reader.close.assert_not_called()
                    else:
                        reader.close.assert_called_once_with()
                else:
                    self.assertEqual(pipeline.run_draw_batch(context, **kwargs), root / "draw_batch_062.json")
                    self.assertEqual(resolver.call_args.kwargs["operation"], pipeline.DRAW)
                    self.assertEqual((resolver.call_args.kwargs["source"], resolver.call_args.kwargs["target"]), (None, None))

    def test_finalizer_requires_all_63_batches_and_closed_workers_before_one_writer(self):
        for failure in (None, "workers", "batch"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                reader, context = reader_fixture(), object()
                budget = SimpleNamespace(allocation={"job_id": "fixture", "hostname": "compute"})
                closed = mock.Mock(return_value=None,
                    side_effect=PermissionError("workers still live") if failure == "workers" else None)
                stack.enter_context(mock.patch.object(pipeline.assembly, "_require_reader_context"))
                opener = stack.enter_context(mock.patch.object(pipeline.staged, "open_decoded_point", return_value=reader))
                visited = []
                primary = SimpleNamespace(point=reader.point)
                incoming, dyadic = object(), object()

                def read(path, **kwargs):
                    reader.close.assert_called_once_with()
                    index = kwargs["expected_batch_index"]
                    self.assertEqual(path, root / "batches" / f"draw_batch_{index:03d}.json")
                    self.assertEqual(kwargs["expected_image_ids"], IDS)
                    self.assertEqual(kwargs["expected_stage_identity"], reader.stage_identity)
                    if failure == "batch" and index == 37:
                        raise FileNotFoundError("one canonical batch missing")
                    visited.append(index)
                    return index

                def assemble(point, batches, *, check):
                    self.assertIs(point, reader.point)
                    self.assertEqual(tuple(batches), tuple(range(63)))
                    check()
                    return primary

                def finalize(stage, final, actual_primary, actual_incoming, actual_dyadic, **kwargs):
                    self.assertEqual(visited, list(range(63)))
                    self.assertIs(actual_primary, primary)
                    self.assertIs(actual_incoming, incoming)
                    self.assertIs(actual_dyadic, dyadic)
                    self.assertGreaterEqual(closed.call_count, 2)
                    kwargs["require_readers_closed"]()
                    self.assertEqual((stage, final), (root / "stage.h5", root / "final.h5"))
                    return "complete-requested-report"

                read_call = stack.enter_context(mock.patch.object(pipeline.artifacts, "read_draw_batch", side_effect=read))
                join = stack.enter_context(mock.patch.object(pipeline.batches, "assemble_bootstrap", side_effect=assemble))
                inc = stack.enter_context(mock.patch.object(pipeline.secondary, "target_incoming", return_value=incoming))
                dyad = stack.enter_context(mock.patch.object(pipeline.secondary, "dyadic_bootstrap", return_value=dyadic))
                publish = stack.enter_context(mock.patch.object(pipeline.staged, "finalize_decoded_point", side_effect=finalize))
                kwargs = dict(require_admitted_context=lambda *args, **kwargs: None, budget=budget,
                    stage_h5=root / "stage.h5", final_h5=root / "final.h5", batch_directory=root / "batches",
                    maximum_collection_bytes=123456, maximum_batch_bytes=12345, require_readers_closed=closed)
                if failure:
                    with self.assertRaises(PermissionError if failure == "workers" else FileNotFoundError):
                        pipeline.run_finalizer(context, **kwargs)
                    publish.assert_not_called(); inc.assert_not_called(); dyad.assert_not_called()
                    if failure == "workers":
                        opener.assert_not_called(); join.assert_not_called(); read_call.assert_not_called()
                else:
                    self.assertEqual(pipeline.run_finalizer(context, **kwargs), "complete-requested-report")
                    self.assertEqual(read_call.call_count, 63)
                    publish.assert_called_once()
                    self.assertIs(inc.call_args.args[0], reader.point)
                    self.assertIs(dyad.call_args.args[0], reader.point)


if __name__ == "__main__":
    unittest.main()
