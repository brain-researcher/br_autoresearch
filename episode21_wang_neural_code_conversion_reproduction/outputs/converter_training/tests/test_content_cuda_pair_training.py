"""Two dormant composition fixtures; no GPU, empirical values or fitting.

Reuse the existing tiny canonical fixture, substituting only shape/axis metadata
for the fixed large pair. The real driver is trapped or observed, never run on
those stand-ins. Execution belongs to a coordinator-selected Slurm allocation.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import content_pair_cuda_training as production
import test_content_pair_driver as driver_fixtures


class ContentCUDAPairTrainingTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with driver_fixtures.ContentPairDriverTests().fixture() as base:
            interface = production.driver.interface
            source = replace(base.source,
                axis=replace(base.source.axis, eligible_native_ids=range(12897)),
                normalized_X=SimpleNamespace(shape=(4712, 12897)))
            target_axis = replace(base.panel.axes[0], eligible_native_ids=range(12294))
            pair = interface.ContentPair(base.panel.manifest, source,
                interface.assembly.required_state("sub-01"), base.normalizations["sub-01"], target_axis)
            initializations = list(base.panel.initializations)
            initializations[4] = replace(initializations[4], input_width=12897,
                                         hidden_width=12897 // 2, output_width=12294)
            panel = replace(base.panel, initializations=tuple(initializations))
            context = SimpleNamespace(operations={production.OPERATION})
            requests = []

            def resolver(supplied, **request):
                self.assertIs(supplied, context)
                self.assertEqual((request["source"], request["target"]), ("sub-03", "sub-01"))
                self.assertIs(request["budget"], base.budget)
                requests.append(request)
                if request["operation"] not in context.operations:
                    raise PermissionError("full production is not admitted")

            correction = production.inputs.LossCorrectionEvidence(
                production.inputs.CORRECTION_RELATIVE, "46881096", production.inputs.CORRECTED_LOSS,
                "fixture_numpy", "fixture_torch")
            identity = (base.budget.allocation["job_id"], base.budget.allocation["hostname"])
            bound = production.inputs.BoundContentInputs(context, resolver, base.budget, identity,
                                                        correction, panel, pair)
            with mock.patch.object(production.inputs, "_read_loss_correction", return_value=correction) as evidence:
                yield SimpleNamespace(base=base, bound=bound, context=context, requests=requests,
                                      evidence=evidence)

    def run_pair(self, fixture, *, bound=None, **changes):
        options = dict(device="cuda:0", block_units=256, row_chunk_size=512, backend="sparse",
                       final_path=fixture.base.path, maximum_bytes=1024**3)
        options.update(changes)
        return production.train_and_publish_cached_cuda_full_pair(
            fixture.bound if bound is None else bound, **options)

    def test_measurement_authority_or_changed_scope_cannot_start_production(self):
        for fault in ("measurement_only", "CPU", "other_pair", "missing_rows", "target_axis",
                      "changed_initialization", "moved_allocation", "changed_correction"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                bound, changes = fixture.bound, {}
                if fault == "measurement_only":
                    fixture.context.operations = {"bind_content_inputs", "measure_content_pair_cuda_cached_four_updates"}
                elif fault == "CPU":
                    changes["device"] = "cpu"
                elif fault == "other_pair":
                    bound = replace(bound, pair=replace(bound.pair,
                        target_axis=replace(bound.pair.target_axis, participant="sub-05")))
                elif fault == "missing_rows":
                    bound = replace(bound, pair=replace(bound.pair, source=replace(bound.source,
                        normalized_X=SimpleNamespace(shape=(4711, 12897)))))
                elif fault == "target_axis":
                    bound = replace(bound, pair=replace(bound.pair,
                        target_axis=replace(bound.pair.target_axis, eligible_native_ids=range(12293))))
                elif fault == "changed_initialization":
                    initial = list(bound.panel.initializations)
                    initial[4] = replace(initial[4], pair_index=5)
                    bound = replace(bound, panel=replace(bound.panel, initializations=tuple(initial)))
                elif fault == "moved_allocation":
                    bound = replace(bound, allocation_identity=("different_job", "different_host"))
                else:
                    fixture.evidence.return_value = replace(bound.loss_correction, loss="mean_not_sum")
                with mock.patch.object(production.driver, "train_and_publish_content_pair") as driver, \
                        mock.patch.object(production.cache, "make_cached_provider") as factory:
                    with self.assertRaises(PermissionError if fault == "measurement_only" else ValueError):
                        self.run_pair(fixture, bound=bound, **changes)
                    driver.assert_not_called()
                    factory.assert_not_called()
                if fault != "changed_correction":
                    fixture.evidence.assert_not_called()
                self.assertFalse(fixture.base.path.exists())

    def test_same_bound_options_fixed_factory_and_original_failure_propagation(self):
        with self.fixture() as fixture:
            bound = fixture.bound
            callback = mock.Mock()
            final_result = object()
            with mock.patch.object(production.driver, "train_and_publish_content_pair", return_value=final_result) as driver:
                self.assertIs(self.run_pair(fixture, on_iteration=callback), final_result)
                expected = dict(device="cuda:0", block_units=256, row_chunk_size=512,
                    backend="sparse", final_path=fixture.base.path, maximum_bytes=1024**3)
                driver.assert_called_once_with(bound.panel, bound.source, target="sub-01", budget=bound.budget,
                    on_iteration=callback, provider_factory=production.cache.make_cached_provider, **expected)
                self.assertEqual(len(fixture.requests), 1)
                request = fixture.requests[0]
                self.assertEqual(request["operation"], "train_and_publish_cached_cuda_full_pair")
                self.assertEqual(dict(request["execution_options"]), {**expected,
                    "feature_read_mode": "bounded_channel_two_pass_cache",
                    "maximum_channel_bytes": 1073741824, "maximum_retained_bytes": 4294967296})
                with self.assertRaises(TypeError):
                    request["execution_options"]["maximum_retained_bytes"] = 1
                callback.assert_not_called()
                for error in (TimeoutError("bounded cancellation"), OSError("final publication failed")):
                    driver.reset_mock()
                    driver.side_effect = error
                    with self.assertRaises(type(error)) as observed:
                        self.run_pair(fixture)
                    self.assertIs(observed.exception, error)
                    driver.assert_called_once()
                self.assertFalse(fixture.base.path.exists())


if __name__ == "__main__":
    unittest.main()
