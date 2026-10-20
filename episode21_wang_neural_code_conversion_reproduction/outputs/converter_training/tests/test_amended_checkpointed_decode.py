"""Stdlib-only composition controls, not scientific or native-HDF integration.

The supplied dependency object has the full roster sizes but fabricates no
scientific scores or HDF payloads. Nine separate lifecycle fixtures own native
close/fsync coverage. Run only in coordinator-owned compute.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import amended_checkpointed_decode as composition
from amended_decode_lifecycle import DecodeLifecycle, DecodeWorkStopped


class CompositionControls(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"SLURM_JOB_ID": "60000002"})
        patcher.start()
        self.addCleanup(patcher.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now, self.events, self.stop_at = 100., [], None
        self.lifecycle = DecodeLifecycle(job_id="60000002", collection_path=self.root / "new.h5",
            record_path=self.root / "closure.json", work_stop_epoch=110., hard_stop_epoch=120.,
            clock=lambda: self.now)
        self.context = object()
        self.budget = SimpleNamespace(allocation={"job_id": "60000002", "hostname": "fixture"},
                                      check=self.lifecycle.check_work)
        self.people = tuple("owner%d" % index for index in range(5))
        self.layers = tuple("layer%d" % index for index in range(19))
        self.streams = tuple((method, source, target) for source in self.people
            for method, target in [("within_source", None)] + [(method, target)
                for target in self.people if target != source for method in ("content_loss", "brain_loss")])
        self.fitted = {stream: "fixture:%s" % (stream,) for stream in self.streams}
        self.dependencies = self.dependencies_fixture()

    def event(self, name):
        self.events.append(name)
        if self.stop_at == name:
            self.now = 110.

    def resolver(self, context, **kwargs):
        self.assertIs(context, self.context)
        self.assertIs(kwargs["budget"], self.budget)
        if kwargs["operation"] == composition.OPERATION:
            self.assertEqual(dict(kwargs["execution_options"]), dict(
                collection_path=self.lifecycle.collection_path, maximum_bytes=1024,
                model_policy=composition.MODEL_POLICY, work_stop_epoch=110., hard_stop_epoch=120.,
                closure_record_path=self.lifecycle.record_path))
        else:
            self.assertIsNone(kwargs["execution_options"])

    def dependencies_fixture(self):
        test = self
        bound = SimpleNamespace(image_ids=tuple("image%d" % index for index in range(224)),
            feature_identity={"fixture": True}, truth_reader=object(), close=lambda: test.event("inputs_closed"))

        def admission(context, resolver, budget, operation, options):
            def check():
                resolver(context, operation=operation, source=None, target=None,
                         budget=budget, execution_options=options)
            check()
            return check

        def manifest(path, *, budget):
            test.event("manifest")
            budget.check()
            return "fixture_manifest"

        def metadata(manifest, budget, check):
            test.event("metadata")
            check()
            return {owner: SimpleNamespace(participant=owner) for owner in test.people}, "training_ids"

        def bind(context, *, require_admitted_context, budget):
            test.event("bind_inputs")
            def check():
                require_admitted_context(context, operation="bind_full_finding_a_inputs",
                    source=None, target=None, budget=budget, execution_options=None)
            bound.check = check
            bound.resolver = require_admitted_context
            check()
            return bound

        def prepare(bound, axes, training, *, authenticated_tau_train_image_ids,
                    execution_context, require_admitted_context, budget):
            test.assertIs(bound.resolver, require_admitted_context)
            test.assertEqual(authenticated_tau_train_image_ids, ("authenticated",))
            test.event("prepare")
            def check():
                require_admitted_context(execution_context, operation="prepare_amended_full_finding_a_models",
                    source=None, target=None, budget=budget, execution_options=None)
            check()
            return SimpleNamespace(check=check, model_policy=composition.MODEL_POLICY,
                unavailable_reasons={}, fitted_object_ids=test.fitted,
                prepared={stream: SimpleNamespace(decoder_axis=axes[stream[1] if stream[2] is None else stream[2]])
                          for stream in test.streams})

        def blocks(manifest, axis, layer, budget, check):
            test.event("decoder_iteration")
            check()
            yield "fixture_block"
            check()

        class Writer:
            def __init__(self, path, **kwargs):
                test.event("writer_constructed")
                self._h5, self.partial, self._closed = None, Path(str(path) + ".partial"), False
                self.check = kwargs["require_admitted"]
                self.image_ids = kwargs["image_ids"]
                self.groups = []
                self._check()

            def _check(self):
                self.check()

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                self.close()

            def close(self):
                if not self._closed:
                    test.event("writer_closed")
                    self._closed = True

            def write_owner_layer(self, *, owner, layer, prepared_by_stream, blocks):
                self._check()
                test.assertEqual(len(prepared_by_stream), 9)
                test.assertEqual(list(blocks), ["fixture_block"])
                self.groups.append((layer, owner))

            def seal_decoded(self):
                self._check()
                test.assertEqual(self.groups, [(layer, owner) for layer in test.layers for owner in test.people])
                test.event("sealed_855")

            def block_provider(self, **kwargs):
                self._check()
                test.assertIs(kwargs["truth_provider"], bound.truth_reader)
                return "fixture_metric_block"

            def result_sink(self, score):
                self._check()
                test.assertEqual(score, "fixture_score")

            def publish_decoded_point(self, point):
                test.event("publish")
                self._check()
                test.assertEqual(point, "fixture_point")
                return (test.lifecycle.collection_path, test.root / "point.json")

        def evaluate(provider, training, image_ids, *, check, result_sink):
            test.event("point_scoring")
            check()
            test.assertEqual(len(image_ids), 224)
            for layer in test.layers:
                for method, source, target in test.streams:
                    test.assertEqual(provider(method=method, source=source, target=target, layer=layer),
                                     "fixture_metric_block")
                    result_sink("fixture_score")
            return "fixture_point"

        return SimpleNamespace(MODEL_POLICY=composition.MODEL_POLICY, require=composition._require,
            original=SimpleNamespace(_admission=admission),
            collection=SimpleNamespace(_require_file_limit=lambda maximum: test.event("file_policy"),
                LAYERS=test.layers, PARTICIPANTS=test.people, STREAMS=test.streams),
            assembly=SimpleNamespace(BASE=Path("/fixture"), MANIFEST="manifest", load_complete_manifest=manifest),
            _decoder_metadata=metadata, _bind_inputs=bind, _prepare_models=prepare,
            _expected_fitted=lambda: dict(test.fitted), _decoder_blocks=blocks,
            publication=SimpleNamespace(AmendedStagedCollection=Writer),
            inference=SimpleNamespace(evaluate_panel=evaluate),
            _result=lambda phase, paths: SimpleNamespace(phase=phase, artifacts=paths))

    def run_composition(self, **kwargs):
        return composition.run_decoded_point(self.context, require_admitted_context=self.resolver,
            budget=self.budget, stage_h5=self.lifecycle.collection_path, maximum_bytes=1024,
            authenticated_tau_train_image_ids=("authenticated",), lifecycle=self.lifecycle, **kwargs)

    def test_complete_sequence_uses_full_roster_before_point_and_publication(self):
        result = self.run_composition(dependencies=self.dependencies)
        self.assertEqual(result.phase, "decode")
        self.assertEqual(len(self.streams), 45)
        self.assertEqual(self.events.count("decoder_iteration"), 95)
        self.assertLess(self.events.index("sealed_855"), self.events.index("point_scoring"))
        self.assertLess(self.events.index("point_scoring"), self.events.index("publish"))
        self.assertEqual(self.events[-2:], ["writer_closed", "inputs_closed"])
        self.assertTrue(self.lifecycle.record_published)
        self.assertFalse(json.loads(self.lifecycle.record_path.read_text())["hdf_close_completed"])

    def test_preparation_deadline_stops_before_writer_and_closes_inputs(self):
        self.stop_at = "prepare"
        with self.assertRaises(DecodeWorkStopped):
            self.run_composition(dependencies=self.dependencies)
        self.assertNotIn("writer_constructed", self.events)
        self.assertEqual(self.events[-1], "inputs_closed")
        self.assertTrue(self.lifecycle.record_published)

    def test_decoder_deadline_unwinds_without_scoring_or_publication(self):
        self.stop_at = "decoder_iteration"
        with self.assertRaises(DecodeWorkStopped):
            self.run_composition(dependencies=self.dependencies)
        self.assertNotIn("sealed_855", self.events)
        self.assertNotIn("point_scoring", self.events)
        self.assertNotIn("publish", self.events)
        self.assertEqual(self.events[-2:], ["writer_closed", "inputs_closed"])

    def test_import_failure_is_inside_outer_lifecycle(self):
        def refused_import():
            self.assertTrue(self.lifecycle._running)
            raise ImportError("fixture dependency unavailable")
        with mock.patch.object(composition, "_load_dependencies", side_effect=refused_import):
            with self.assertRaises(ImportError):
                self.run_composition()
        self.assertTrue(self.lifecycle.record_published)
        self.assertEqual(json.loads(self.lifecycle.record_path.read_text())["operation_error_type"], "ImportError")

    def test_denied_admission_precedes_scientific_import(self):
        def denied(context, **kwargs):
            self.assertTrue(self.lifecycle._running)
            raise PermissionError("fixture operation not admitted")
        with mock.patch.object(self, "resolver", side_effect=denied), \
                mock.patch.object(composition, "_load_dependencies", side_effect=AssertionError("imported before admission")) as load:
            with self.assertRaises(PermissionError):
                self.run_composition()
        load.assert_not_called()
        self.assertTrue(self.lifecycle.record_published)


if __name__ == "__main__":
    unittest.main()
