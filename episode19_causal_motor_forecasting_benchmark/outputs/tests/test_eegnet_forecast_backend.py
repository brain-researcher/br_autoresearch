"""Six focused EEGNet checks; coordinator-owned Slurm execution only.

Operator fixtures are implementation checks, not simulated scientific results.
One five-example, 125-sample actual 100-epoch fit checks backend integration.
"""
from dataclasses import replace
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import eegnet_forecast_backend as backend
from classical_forecast_backend import (
    FittingExample, FittingScope, GroupIdentity, PastWindow, UnresolvedFit,
    normalized_log_probabilities,
)
from forecast_components import FactorizedBlocks

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


class EEGNetBackendTests(unittest.TestCase):
    def scope(self, source="WAY"):
        return FittingScope.development(REGISTRY, source=source,
            fold_id="WAY.P1.validation_series6" if source == "WAY" else "self_paced.f0",
            participant=1 if source == "WAY" else None)

    def examples(self, source="WAY"):
        scope = self.scope(source)
        groups = scope.allowed_series if source == "WAY" else scope.allowed_participants
        offset = normalized_log_probabilities(np.linspace(-2., 1., 19)[None])[0]
        examples = []
        for index, group in enumerate(groups):
            person, series = (1, group) if source == "WAY" else (group, "run-1")
            boundary = f"{source}-{person}-{series}"
            window = PastWindow(np.arange(125 * 29).reshape(125, 29) / 1000. + index / 10.,
                                10. - np.arange(125, 0, -1) / 125., [boundary] * 125)
            examples.append(FittingExample(window, 10., GroupIdentity(source, person, series, boundary),
                                           18, 18, offset.copy()))
        return examples

    def semantics(self, source="WAY", **changes):
        result = dict(registry=REGISTRY, fitting_scope=self.scope(source), analysis_sample_rate_hz=125.,
            channel_order=backend.CHANNELS, adapter="original", budget_check=lambda: None)
        result.update(changes)
        return result

    def fit(self, examples=None, *, source="WAY", model=None, **changes):
        model = model or backend.EEGNetForecastModel()
        report = model.fit([self.examples(source) if examples is None else examples],
            recipe=backend.eegnet_recipes(REGISTRY)[0], seed=1901, semantics=self.semantics(source, **changes))
        return model, report

    @staticmethod
    def positive_weights(network):
        with torch.no_grad():
            network.temporal.weight.fill_(1. / network.temporal_samples)
            network.spatial.weight.fill_(1. / 29.)
            network.depthwise.weight.fill_(1. / network.separable_samples)
            network.pointwise.weight.fill_(1. / network.channels)
            for module in (network.temporal, network.spatial, network.depthwise, network.pointwise):
                module.bias.zero_()

    @staticmethod
    def scalar_oracle(values, *, start, total, temporal, separable):
        # Independent direct finite sums in raw/coarse sample coordinates; fixed
        # left-zero averages retain full denominators at both slot boundaries.
        temporal_values = np.asarray([sum(values[max(0, end - temporal + 1):end + 1]) / temporal
                                      for end in range(len(values))])
        first_edges = [end for end in range(start, start + len(values)) if (total - 1 - end) % 4 == 0]
        coarse = np.asarray([sum(temporal_values[max(0, end - start - 3):end - start + 1]) / 4.
                             for end in first_edges])
        separable_values = np.asarray([sum(coarse[max(0, end - separable + 1):end + 1]) / separable
                                       for end in range(len(coarse))])
        final = [(index, edge) for index, edge in enumerate(first_edges) if (total - 1 - edge) % 32 == 0]
        return (np.asarray([sum(separable_values[max(0, index - 7):index + 1]) / 8. for index, _ in final]),
                np.asarray([edge for _, edge in final]))

    def test_registered_kernels_compact_channels_and_boundary_average(self):
        self.assertEqual(backend.kernel_samples(500.), (64, 8))
        self.assertEqual(backend.kernel_samples(250.), (32, 4))
        self.assertEqual(backend.kernel_samples(1000.), (128, 16))
        self.assertEqual(backend.kernel_samples(125.), (16, 2))
        for invalid in (0., -1., 62.6, 256., np.nan):
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                backend.kernel_samples(invalid)
        with torch.random.fork_rng(devices=[]):
            for filters in (4, 8):
                network = backend.EEGNetCorrectionNetwork(filters, analysis_sample_rate_hz=500.,
                    recent_samples=100, adapter="original")
                self.assertEqual(network.temporal.kernel_size, (1, 64))
                self.assertEqual(network.spatial.kernel_size, (29, 1))
                self.assertEqual(network.spatial.groups, filters)
                self.assertEqual(network.spatial.out_channels, 2 * filters)
                self.assertEqual(network.depthwise.kernel_size, (8,))
                self.assertEqual(network.depthwise.groups, 2 * filters)
                self.assertEqual(network.pointwise.kernel_size, (1,))
                self.assertEqual((network.readout.in_features, network.readout.out_features), (4 * filters, 19))
                self.assertEqual(sum(isinstance(module, torch.nn.Dropout) for module in network.modules()), 2)
                self.assertFalse(any(isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm)) for module in network.modules()))
                self.assertTrue(all(parameter.dtype == torch.float32 and parameter.device.type == "cpu"
                                    for parameter in network.parameters()))
        torch.testing.assert_close(backend._past_average(torch.ones((1, 1, 5)), 4),
                                  torch.tensor([[[.25, .5, .75, 1., 1.]]]), rtol=0, atol=0)

    def test_independent_operator_oracle_and_future_mutation(self):
        with torch.random.fork_rng(devices=[]):
            network = backend.EEGNetCorrectionNetwork(4, analysis_sample_rate_hz=500.,
                recent_samples=100, adapter="original").eval()
            self.positive_weights(network)
            raw = np.linspace(.1, 1., 500)
            values = torch.tensor(np.tile(raw[None, :, None], (1, 1, 29)), dtype=torch.float32)
            expected, expected_edges = self.scalar_oracle(raw, start=0, total=500, temporal=64, separable=8)
            with torch.no_grad():
                hidden, edges = network.encoded(values)
                np.testing.assert_array_equal(edges.numpy(), np.arange(19, 500, 32))
                np.testing.assert_array_equal(edges.numpy(), expected_edges)
                np.testing.assert_allclose(hidden.numpy()[0], np.tile(expected, (8, 1)), atol=2e-7, rtol=2e-6)
                future = values.clone()
                future[:, 310:] += 100.
                future_hidden, _ = network.encoded(future)
                torch.testing.assert_close(hidden[:, :, edges < 310], future_hidden[:, :, edges < 310], rtol=0, atol=0)
                self.assertGreater(float((hidden[:, :, edges >= 310] - future_hidden[:, :, edges >= 310]).abs().max()), 0.)

    def test_factorial_phase_boundary_masks_and_equal_capacity(self):
        with torch.random.fork_rng(devices=[]):
            original = backend.EEGNetCorrectionNetwork(4, analysis_sample_rate_hz=500.,
                recent_samples=100, adapter="original").eval()
            self.positive_weights(original)
            factorial = backend.EEGNetCorrectionNetwork(4, analysis_sample_rate_hz=500.,
                recent_samples=100, adapter="factorial").eval()
            factorial.load_state_dict(original.state_dict())
            raw = np.linspace(.1, 1., 500)
            values = torch.tensor(np.tile(raw[None, :, None], (1, 1, 29)), dtype=torch.float32)
            with torch.no_grad():
                full, edges = original.encoded(values)
                old, old_edges = factorial.encoded(values[:, :400], full_window_samples=500)
                recent, recent_edges = factorial.encoded(values[:, 400:], start_sample=400, full_window_samples=500)
                np.testing.assert_array_equal(old_edges.numpy(), edges.numpy()[edges.numpy() < 400])
                np.testing.assert_array_equal(recent_edges.numpy(), [403, 435, 467, 499])
                torch.testing.assert_close(old, full[:, :, edges < 400], atol=2e-7, rtol=2e-6)
                expected, expected_edges = self.scalar_oracle(raw[400:], start=400, total=500, temporal=64, separable=8)
                np.testing.assert_array_equal(recent_edges.numpy(), expected_edges)
                np.testing.assert_allclose(recent.numpy()[0], np.tile(expected, (8, 1)), atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(factorial.summaries(values), torch.cat((old.mean(2), recent.mean(2)), 1))
                changed_older = values.clone()
                changed_older[:, :400] += 10.
                torch.testing.assert_close(factorial.summaries(values)[:, 8:],
                    factorial.summaries(changed_older)[:, 8:], rtol=0, atol=0)
                self.assertGreater(float((original.summaries(values)[:, 8:]
                    - original.summaries(changed_older)[:, 8:]).abs().max()), 0.)
                blocks = FactorizedBlocks(values[0, :400].numpy(), np.ones((400, 29)) / 7., values[0, 400:].numpy())
                def prediction(block, r, o):
                    return factorial(torch.tensor(block.reconstruct(r, o)[None], dtype=torch.float32))
                for o in (0, 1):
                    torch.testing.assert_close(prediction(blocks, 0, o),
                        prediction(replace(blocks, recent=blocks.recent + 1000.), 0, o), rtol=0, atol=0)
                for r in (0, 1):
                    torch.testing.assert_close(prediction(blocks, r, 0),
                        prediction(replace(blocks, older_slow=blocks.older_slow - 200.), r, 0), rtol=0, atol=0)
                before = factorial(values)
                factorial(changed_older)
                torch.testing.assert_close(before, factorial(values), rtol=0, atol=0)
            self.assertEqual([(name, tuple(value.shape)) for name, value in original.named_parameters()],
                             [(name, tuple(value.shape)) for name, value in factorial.named_parameters()])

    def test_recipe_source_grid_and_complete_window_admission(self):
        recipes = backend.eegnet_recipes(REGISTRY)
        self.assertEqual(len(recipes), 12)
        modified = copy.deepcopy(REGISTRY)
        modified["reference_features"]["EEGNet_pool_widths"] = [8, 4]
        with self.assertRaises(ValueError):
            backend.eegnet_recipes(modified)
        def unopened():
            self.fail("invalid semantics must not consume fitting windows")
            yield []
        cases = [self.semantics(analysis_sample_rate_hz=256.), self.semantics(analysis_sample_rate_hz=62.5),
                 self.semantics(eegnet_summary="last"), self.semantics(eegnet_boundary="cached"),
                 self.semantics(eegnet_pool_phase="slot_end"), self.semantics(channel_order=backend.CHANNELS[::-1]),
                 self.semantics(fitting_scope=FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9)))]
        absent = self.semantics()
        del absent["analysis_sample_rate_hz"]
        cases.append(absent)
        examples = self.examples()
        first = examples[0]
        windows = [replace(first.past_input, values=first.past_input.values[:-1]),
                   replace(first.past_input, sample_times=first.past_input.sample_times + .008),
                   replace(first.past_input, boundary_ids=["other", *first.past_input.boundary_ids[1:]])]
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            for semantics in cases:
                with self.subTest(semantics=semantics), self.assertRaises((ValueError, KeyError)):
                    backend.EEGNetForecastModel().fit(unopened(), recipe=recipes[0], seed=1901, semantics=semantics)
            for window in windows:
                with self.subTest(window=window), self.assertRaises(ValueError):
                    self.fit([replace(first, past_input=window), *examples[1:]])
            model = backend.EEGNetForecastModel()
            with self.assertRaisesRegex(UnresolvedFit, "no_observed_EEGNet"):
                self.fit([replace(item, observed_category=-1, complete_bins=0) for item in examples], model=model)
            self.assertFalse(model.fitted)
            forbidden.assert_not_called()

    def test_global_person_weights_fixed_offsets_and_masked_zeros(self):
        examples = self.examples("self_paced")
        zero = replace(examples[0], past_input=replace(examples[0].past_input, values=np.zeros((125, 29))))
        examples = [zero, examples[0], *examples[1:]]
        supplied = np.asarray([item.past_input.values for item in examples]).copy()
        offsets = np.asarray([item.context_log_probabilities for item in examples]).copy()
        with mock.patch.object(backend, "_train_fixed_schedule", return_value=(object(), 100, 1.)) as train:
            _, report = self.fit(examples, source="self_paced", adapter="factorial")
        np.testing.assert_array_equal(train.call_args.args[0], supplied)
        np.testing.assert_array_equal(train.call_args.args[0][0], 0.)
        np.testing.assert_array_equal(train.call_args.args[1], offsets)
        np.testing.assert_allclose(train.call_args.args[4][:2], 1 / 24)
        np.testing.assert_allclose(train.call_args.args[4][2:], 1 / 12)
        self.assertEqual(train.call_args.kwargs["adapter"], "factorial")
        self.assertEqual(report["temporal_kernel_raw_samples"], 16)
        self.assertEqual(report["separable_kernel_post_pool4_samples"], 2)
        self.assertFalse(report["source_admission_established_by_backend"])
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            with self.assertRaisesRegex(UnresolvedFit, "required fitting participant"):
                self.fit(examples[:-1], source="self_paced")
            forbidden.assert_not_called()

    def test_actual_fixed_schedule_budget_frozen_inference_and_zero_correction(self):
        self.assertEqual(torch.get_num_threads(), 1, "coordinator must set one numerical thread")
        events, defaults = [], []
        actual_step = torch.optim.AdamW.step
        def step(optimizer, *args, **kwargs):
            self.assertEqual(events[-1], "check")
            events.append("step")
            if not defaults:
                defaults.append(optimizer.defaults.copy())
            return actual_step(optimizer, *args, **kwargs)
        rng = torch.random.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        examples = self.examples()
        examples[0] = replace(examples[0], observed_category=0, complete_bins=0)
        examples[1] = replace(examples[1], observed_category=-1, complete_bins=9)
        offsets_before = np.asarray([item.context_log_probabilities for item in examples]).copy()
        with mock.patch.object(torch.optim.AdamW, "step", new=step):
            model, report = self.fit(examples, budget_check=lambda: events.append("check"))
        self.assertEqual(report["epochs"], 100)
        self.assertEqual(report["optimizer_steps"], 100)
        self.assertEqual(events.count("step"), 100)
        self.assertEqual(defaults[0]["lr"], .001)
        self.assertEqual(defaults[0]["weight_decay"], .001)
        self.assertTrue(np.isfinite(report["objective"]))
        self.assertFalse(model.network.training)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in model.network.parameters()))
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn)
        np.testing.assert_array_equal(np.asarray([item.context_log_probabilities for item in examples]), offsets_before)
        first = examples[0]
        model.reset(boundary_id=first.identity.boundary_id)
        prediction = model.predict_log_probabilities(first.past_input, decision_time=first.decision_time,
                                                     context_log_probabilities=first.context_log_probabilities)
        self.assertAlmostEqual(float(np.exp(prediction).sum()), 1., places=12)
        with self.assertRaises(ValueError):
            model.predict_log_probabilities(first.past_input, decision_time=first.decision_time,
                                             context_log_probabilities=first.context_log_probabilities)
        with torch.no_grad():
            model.network.readout.weight.zero_()
            model.network.readout.bias.zero_()
        model.reset(boundary_id=first.identity.boundary_id)
        prediction = model.predict_log_probabilities(first.past_input, decision_time=first.decision_time,
                                                     context_log_probabilities=first.context_log_probabilities)
        np.testing.assert_allclose(prediction, first.context_log_probabilities, atol=1e-12, rtol=0)
        stopped = backend.EEGNetForecastModel()
        with self.assertRaisesRegex(RuntimeError, "allocation ended"):
            self.fit(model=stopped, budget_check=mock.Mock(side_effect=RuntimeError("allocation ended")))
        self.assertFalse(stopped.fitted)


if __name__ == "__main__":
    unittest.main()
