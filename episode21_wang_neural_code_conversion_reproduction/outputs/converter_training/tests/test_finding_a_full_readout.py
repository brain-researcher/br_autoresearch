"""Two full-roster composition checks; run in coordinated Slurm compute only.

Numerical predictors, grouped I/O and original inference have their own
focused checks. These instrument the join, not a synthetic scientific study.
"""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_full_readout as readout


IDS = tuple(f"image-{index:03d}" for index in range(224))
TAU_IDS = tuple(f"tau-train-{index:03d}" for index in range(897))


class FullFindingAReadoutTests(unittest.TestCase):
    def test_denied_admission_and_decoder_reader_failure_precede_outputs(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory).resolve() / "full.h5"
            budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"})
            stack.enter_context(mock.patch.object(readout.assembly, "_require_reader_context"))
            cap = stack.enter_context(mock.patch.object(readout.collection, "_require_file_limit"))
            manifest = stack.enter_context(mock.patch.object(readout.assembly, "load_complete_manifest"))
            binding = stack.enter_context(mock.patch.object(readout.inputs, "bind_full_finding_a_inputs"))
            preparation = stack.enter_context(mock.patch.object(readout.models, "prepare_full_models"))
            writer = stack.enter_context(mock.patch.object(readout.grouped, "GroupedFindingACollection"))
            context = object()
            kwargs = dict(budget=budget, collection_path=path, maximum_bytes=123456,
                          authenticated_tau_train_image_ids=TAU_IDS)
            with self.assertRaises(PermissionError):
                readout.run_full_finding_a_readout(context,
                    require_admitted_context=mock.Mock(side_effect=PermissionError("closed")), **kwargs)
            for operation in (cap, manifest, binding, preparation, writer):
                operation.assert_not_called()
            cap.side_effect = ValueError("fatal cap missing")
            with self.assertRaisesRegex(ValueError, "fatal cap"):
                readout.run_full_finding_a_readout(context,
                    require_admitted_context=lambda *args, **kwargs: None, **kwargs)
            for operation in (manifest, binding, preparation, writer):
                operation.assert_not_called()

        axis = SimpleNamespace(participant="sub-03")
        raw = SimpleNamespace(participant="sub-03", layer="fc8",
                              full_layer_Fortran_index=np.array([0, 1]))
        for wrong in (False, True):
            closed = []

            def supplied():
                try:
                    yield SimpleNamespace(participant="sub-05", layer="fc8") if wrong else raw
                finally:
                    closed.append(True)

            with mock.patch.object(readout.assembly, "iter_sparse_blocks", return_value=supplied()) as iterator, \
                    mock.patch.object(readout.interface, "map_decoder_block", return_value="mapped") as mapper:
                values = readout._decoder_blocks("manifest", axis, "fc8", "budget", lambda: None)
                if wrong:
                    with self.assertRaisesRegex(ValueError, "owner_and_layer"):
                        tuple(values)
                    mapper.assert_not_called()
                else:
                    self.assertEqual(tuple(values), ("mapped",))
                    mapper.assert_called_once_with(raw, axis, channel=None)
                iterator.assert_called_once_with("manifest", participant="sub-03", layer="fc8",
                                                 budget="budget", block_units=256)
            self.assertEqual(closed, [True])

    def test_all_original_slots_grouped_decoding_inference_and_failure_cleanup(self):
        self.assertEqual(len(readout.collection.SLOTS), 855)
        self.assertEqual(readout.batched.original.BOOTSTRAP_DRAWS, 1000)
        self.assertEqual(readout.secondary.DYADIC_DRAWS, 1000)
        case = self
        for failure in (None, "models", "inference"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                path = Path(directory).resolve() / "full.h5"
                context = object()
                budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"})
                axes = {owner: SimpleNamespace(participant=owner) for owner in readout.collection.PARTICIPANTS}
                training = {owner: (f"training-{owner}",) for owner in axes}
                bound = SimpleNamespace(image_ids=IDS, feature_identity={"fixture": "identity"},
                    truth_reader=object(), check=mock.Mock(), close=mock.Mock())
                missing = {stream: "observed_pair4" if stream == ("brain_loss", "sub-03", "sub-01")
                    else "source_inferred_not_executed" for stream in readout.collection.STREAMS
                    if stream[0] == "brain_loss" and "sub-01" in stream[1:]}
                prepared = {stream: SimpleNamespace(decoder_axis=axes[stream[1] if stream[0] == "within_source"
                    else stream[2]]) for stream in readout.collection.STREAMS if stream not in missing}
                fitted = {stream: None if stream in missing else "fixture-fitted-object"
                          for stream in readout.collection.STREAMS}
                models = SimpleNamespace(prepared=prepared, fitted_object_ids=fitted,
                                         unavailable_reasons=missing, check=mock.Mock())
                primary = SimpleNamespace(point=object())
                incoming, dyadic, score = object(), object(), object()
                published = (path, path.with_suffix(".manifest.json"), path.with_suffix(".report.json"))
                writers, reads = [], []

                class Writer:
                    def __init__(self, actual_path, **kwargs):
                        case.assertEqual(actual_path, path)
                        case.assertEqual(kwargs["image_ids"], IDS)
                        case.assertIs(kwargs["fitted_object_ids"], fitted)
                        case.assertIs(kwargs["decoder_axes_by_owner"], axes)
                        case.assertIs(kwargs["decoder_training_image_ids_by_owner"], training)
                        kwargs["require_admitted"]()
                        self.written, self.missing, self.scores = set(), {}, []
                        self.sealed = self.published = False
                        self.exit_type = None
                        writers.append(self)

                    def __enter__(self):
                        return self

                    def __exit__(self, kind, *unused):
                        self.exit_type = kind

                    def write_owner_layer(self, *, owner, layer, prepared_by_stream, blocks):
                        expected = {key for key, value in prepared.items() if value.decoder_axis.participant == owner}
                        case.assertEqual(set(prepared_by_stream), expected)
                        case.assertEqual(tuple(blocks), ((owner, layer),))
                        for key in expected:
                            case.assertNotIn((*key, layer), self.written)
                            self.written.add((*key, layer))

                    def unavailable_layer(self, *, method, source, target, layer, reason):
                        key = method, source, target, layer
                        case.assertNotIn(key, self.missing)
                        self.missing[key] = reason

                    def seal_decoded(self):
                        case.assertEqual((len(self.written), len(self.missing)), (703, 152))
                        case.assertEqual(self.written | set(self.missing), set(readout.collection.SLOTS))
                        case.assertFalse(self.written & set(self.missing))
                        case.assertEqual(Counter(self.missing.values()),
                                         {"observed_pair4": 19, "source_inferred_not_executed": 133})
                        self.sealed = True

                    def block_provider(self, *, method, source, target, layer, truth_provider):
                        case.assertTrue(self.sealed)
                        case.assertIs(truth_provider, bound.truth_reader)
                        return method, source, target, layer

                    def result_sink(self, result):
                        self.scores.append(result)

                    def publish(self, actual_primary, actual_incoming, actual_dyadic):
                        case.assertIs(actual_primary, primary)
                        case.assertIs(actual_incoming, incoming)
                        case.assertIs(actual_dyadic, dyadic)
                        self.published = True
                        return published

                def decoder_blocks(manifest, axis, layer, actual_budget, check):
                    self.assertIs(actual_budget, budget)
                    check()
                    reads.append((axis.participant, layer))
                    yield axis.participant, layer

                def bootstrap(provider, actual_training, images, *, check, result_sink):
                    self.assertIs(actual_training, training)
                    self.assertEqual(images, IDS)
                    self.assertTrue(writers[0].sealed)
                    check()
                    if failure == "inference":
                        raise TimeoutError("fixture inference deadline")
                    self.assertEqual(provider(method="content_loss", source="sub-01", target="sub-03", layer="fc8"),
                                     ("content_loss", "sub-01", "sub-03", "fc8"))
                    result_sink(score)
                    return primary

                def resolver(actual, **kwargs):
                    self.assertIs(actual, context)
                    self.assertEqual(kwargs["operation"], readout.OPERATION)
                    self.assertEqual((kwargs["source"], kwargs["target"]), (None, None))
                    self.assertIs(kwargs["budget"], budget)
                    self.assertEqual(dict(kwargs["execution_options"]),
                                     {"collection_path": path, "maximum_bytes": 123456})

                stack.enter_context(mock.patch.object(readout.assembly, "_require_reader_context"))
                stack.enter_context(mock.patch.object(readout.collection, "_require_file_limit"))
                stack.enter_context(mock.patch.object(readout.assembly, "load_complete_manifest", return_value="manifest"))
                stack.enter_context(mock.patch.object(readout.first, "_decoder_metadata", return_value=(axes, training)))
                binding = stack.enter_context(mock.patch.object(readout.inputs, "bind_full_finding_a_inputs", return_value=bound))
                prepare = stack.enter_context(mock.patch.object(readout.models, "prepare_full_models", return_value=models,
                    side_effect=FileNotFoundError("expected model absent") if failure == "models" else None))
                constructor = stack.enter_context(mock.patch.object(readout.grouped, "GroupedFindingACollection", Writer))
                stack.enter_context(mock.patch.object(readout, "_decoder_blocks", side_effect=decoder_blocks))
                boot = stack.enter_context(mock.patch.object(readout.batched, "bootstrap_panel", side_effect=bootstrap))
                inc = stack.enter_context(mock.patch.object(readout.secondary, "target_incoming", return_value=incoming))
                dyad = stack.enter_context(mock.patch.object(readout.secondary, "dyadic_bootstrap", return_value=dyadic))
                kwargs = dict(require_admitted_context=resolver, budget=budget, collection_path=path,
                              maximum_bytes=123456, authenticated_tau_train_image_ids=TAU_IDS)
                if failure:
                    with self.assertRaises(FileNotFoundError if failure == "models" else TimeoutError):
                        readout.run_full_finding_a_readout(context, **kwargs)
                    inc.assert_not_called(); dyad.assert_not_called()
                    if failure == "models":
                        self.assertEqual(writers, [])
                        boot.assert_not_called()
                    else:
                        self.assertFalse(writers[0].published)
                        self.assertIs(writers[0].exit_type, TimeoutError)
                else:
                    self.assertEqual(readout.run_full_finding_a_readout(context, **kwargs), published)
                    self.assertEqual(writers[0].scores, [score])
                    self.assertIs(inc.call_args.args[0], primary.point)
                    self.assertIs(dyad.call_args.args[0], primary.point)
                if failure != "models":
                    self.assertEqual(len(reads), 95)
                    self.assertEqual(set(reads), {(owner, layer) for owner in axes for layer in readout.collection.LAYERS})
                binding.assert_called_once_with(context, require_admitted_context=resolver, budget=budget)
                prepare.assert_called_once_with(bound, axes, training, authenticated_tau_train_image_ids=TAU_IDS,
                    execution_context=context, require_admitted_context=resolver, budget=budget)
                bound.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
