"""Two orchestration checks, with no GPU, empirical inputs or model fit.

Tiny existing canonical fixtures supply object identities; explicit large-axis
stand-ins are used only before a mocked driver. These are not reduced fits.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import content_pair_cuda_roster as production
import test_content_pair_driver as fixtures


class ContentCudaRosterTrainingTests(unittest.TestCase):
    @contextmanager
    def fixture(self, stage):
        source_id, target_id, index, width = production.SCOPES[stage]
        with fixtures.ContentPairDriverTests().fixture() as base:
            interface = production.driver.interface
            source = replace(base.source,
                axis=replace(base.source.axis, eligible_native_ids=range(12897)),
                image_ids=tuple(f"fixture_image_{i}" for i in range(4712)),
                normalized_X=SimpleNamespace(shape=(4712, 12897)))
            target_axis = replace(next(axis for axis in base.panel.axes if axis.participant == target_id),
                                  eligible_native_ids=range(width))
            pair = interface.ContentPair(base.panel.manifest, source,
                interface.assembly.required_state(target_id), base.normalizations[target_id], target_axis)
            initializations = list(base.panel.initializations)
            initializations[index] = replace(initializations[index], input_width=12897,
                                             hidden_width=6448, output_width=width)
            panel = replace(base.panel, initializations=tuple(initializations))
            context = SimpleNamespace(operations={production.OPERATION})
            requests = []

            def resolver(actual, **request):
                self.assertIs(actual, context)
                self.assertIs(request["budget"], base.budget)
                requests.append(request)
                if (request["source"], request["target"]) != (source_id, target_id) \
                        or request["operation"] not in context.operations:
                    raise PermissionError("fixture denies other pair or operation")

            correction = production.inputs.LossCorrectionEvidence(
                production.inputs.CORRECTION_RELATIVE, "46881096", production.inputs.CORRECTED_LOSS,
                "fixture_numpy", "fixture_torch")
            identity = (base.budget.allocation["job_id"], base.budget.allocation["hostname"])
            bound = production.inputs.BoundContentInputs(context, resolver, base.budget,
                                                        identity, correction, panel, pair)
            with mock.patch.object(production.inputs, "_read_loss_correction", return_value=correction) as evidence:
                yield SimpleNamespace(base=base, stage=stage, target=target_id, index=index, width=width,
                                      bound=bound, context=context, requests=requests, evidence=evidence)

    def call(self, f, *, bound=None, **changes):
        options = dict(stage=f.stage, device="cuda:0", block_units=256, row_chunk_size=512,
                       backend="sparse", final_path=f.base.path, maximum_bytes=1024**3)
        options.update(changes)
        return production.train_and_publish_cached_cuda_roster_pair(f.bound if bound is None else bound, **options)

    def test_three_literal_scopes_refuse_wrong_admission_axes_or_initialization_before_driver(self):
        self.assertEqual(dict(production.SCOPES), {
            "roster-005": ("sub-03", "sub-05", 5, 10891),
            "roster-006": ("sub-03", "sub-06", 6, 12023),
            "roster-007": ("sub-03", "sub-07", 7, 12057)})
        for stage in production.SCOPES:
            for fault in ("measurement", "pair4_operation", "unknown_stage", "cross_stage", "CPU",
                          "target_axis", "missing_rows", "missing_images", "initialization", "schedule",
                          "allocation", "correction"):
                with self.subTest(stage=stage, fault=fault), self.fixture(stage) as f:
                    bound, changes = f.bound, {}
                    if fault == "measurement":
                        f.context.operations = {"measure_content_pair_cuda_cached_four_updates"}
                    elif fault == "pair4_operation":
                        f.context.operations = {"train_and_publish_cached_cuda_full_pair"}
                    elif fault == "unknown_stage":
                        changes["stage"] = "roster-004"
                    elif fault == "cross_stage":
                        changes["stage"] = "roster-006" if stage == "roster-005" else "roster-005"
                    elif fault == "CPU":
                        changes["device"] = "cpu"
                    elif fault == "target_axis":
                        bound = replace(bound, pair=replace(bound.pair,
                            target_axis=replace(bound.pair.target_axis, eligible_native_ids=range(f.width - 1))))
                    elif fault in ("missing_rows", "missing_images"):
                        source = replace(bound.source, **({"normalized_X": SimpleNamespace(shape=(4711, 12897))}
                            if fault == "missing_rows" else {"image_ids": bound.source.image_ids[:-1]}))
                        bound = replace(bound, pair=replace(bound.pair, source=source))
                    elif fault == "initialization":
                        states = list(bound.panel.initializations)
                        states[f.index] = replace(states[f.index], pair_index=4)
                        bound = replace(bound, panel=replace(bound.panel, initializations=tuple(states)))
                    elif fault == "schedule":
                        channels = list(bound.panel.channel_schedule.channels_by_layer)
                        channels[0] = channels[0][:-1]
                        bound = replace(bound, panel=replace(bound.panel, channel_schedule=replace(
                            bound.panel.channel_schedule, channels_by_layer=tuple(channels))))
                    elif fault == "allocation":
                        bound = replace(bound, allocation_identity=("other_job", "other_host"))
                    else:
                        f.evidence.return_value = replace(bound.loss_correction, loss="different_loss")
                    with mock.patch.object(production.driver, "train_and_publish_content_pair") as driver, \
                            mock.patch.object(production.cache, "make_cached_provider") as provider:
                        with self.assertRaises(PermissionError if fault in ("measurement", "pair4_operation", "cross_stage")
                                               else ValueError):
                            self.call(f, bound=bound, **changes)
                        driver.assert_not_called()
                        provider.assert_not_called()
                    if fault != "correction":
                        f.evidence.assert_not_called()
                    self.assertFalse(f.base.path.exists())

    def test_exact_original_driver_arguments_and_immutable_options_with_failure_propagation(self):
        for stage in production.SCOPES:
            with self.subTest(stage=stage), self.fixture(stage) as f:
                callback, result = mock.Mock(), object()
                with mock.patch.object(production.driver, "train_and_publish_content_pair", return_value=result) as driver:
                    self.assertIs(self.call(f, on_iteration=callback), result)
                    options = dict(device="cuda:0", block_units=256, row_chunk_size=512, backend="sparse",
                                   final_path=f.base.path, maximum_bytes=1024**3)
                    driver.assert_called_once_with(f.bound.panel, f.bound.source,
                        target=f.target, budget=f.bound.budget, on_iteration=callback,
                        provider_factory=production.cache.make_cached_provider, **options)
                    self.assertEqual(len(f.requests), 1)
                    self.assertEqual(f.requests[0]["operation"], production.OPERATION)
                    self.assertEqual(dict(f.requests[0]["execution_options"]), {**options,
                        "feature_read_mode": "bounded_channel_two_pass_cache",
                        "maximum_channel_bytes": 1073741824, "maximum_retained_bytes": 4294967296})
                    with self.assertRaises(TypeError):
                        f.requests[0]["execution_options"]["backend"] = "dense"
                    callback.assert_not_called()
                    for error in (TimeoutError("cancellation"), OSError("publication failure")):
                        driver.reset_mock()
                        driver.side_effect = error
                        with self.assertRaises(type(error)) as seen:
                            self.call(f)
                        self.assertIs(seen.exception, error)
                        driver.assert_called_once()
                    self.assertFalse(f.base.path.exists())


if __name__ == "__main__":
    unittest.main()
