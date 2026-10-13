"""Five focused TCN checks; run only in a coordinator-owned Slurm allocation.

Tiny array fixtures test the selected operator, not EEG effects or source
timing/calibration. One five-example actual 100-epoch fit checks integration.
Shared censored-loss mathematics is covered by its existing GRU tests.
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
import tcn_forecast_backend as backend
from classical_forecast_backend import (
    FittingExample, FittingScope, GroupIdentity, PastWindow, UnresolvedFit,
    normalized_log_probabilities,
)
from forecast_components import FactorizedBlocks

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


class TCNBackendTests(unittest.TestCase):
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
            window = PastWindow(np.arange(145).reshape(5, 29) / 100. + index / 10.,
                                10. - np.arange(5, 0, -1) / 5., [boundary] * 5)
            examples.append(FittingExample(window, 10., GroupIdentity(source, person, series, boundary),
                                           18, 18, offset.copy()))
        return examples

    def semantics(self, source="WAY", **changes):
        result = dict(registry=REGISTRY, fitting_scope=self.scope(source), analysis_sample_rate_hz=5.,
            channel_order=backend.CHANNELS, adapter="original", budget_check=lambda: None)
        result.update(changes)
        return result

    def fit(self, examples=None, *, source="WAY", model=None, **changes):
        model = model or backend.TCNForecastModel()
        report = model.fit([self.examples(source) if examples is None else examples],
            recipe=backend.tcn_recipes(REGISTRY)[0], seed=1901, semantics=self.semantics(source, **changes))
        return model, report

    @staticmethod
    def positive_weights(network):
        with torch.no_grad():
            for module in network.modules():
                if isinstance(module, torch.nn.Conv1d):
                    module.weight.fill_(1. / module.in_channels)
                    module.bias.zero_()

    def test_registered_residual_operator_has_exact_past_only_125_sample_support(self):
        with torch.random.fork_rng(devices=[]):
            for width in (16, 32):
                network = backend.TCNCorrectionNetwork(width, recent_samples=2, adapter="original").eval()
                self.positive_weights(network)
                self.assertEqual(len(network.blocks), 5)
                self.assertEqual(network.readout.in_features, 2 * width)
                self.assertEqual(network.readout.out_features, 19)
                self.assertEqual([block.first.dilation for block in network.blocks], [(d,) for d in backend.DILATIONS])
                self.assertTrue(all(block.first.kernel_size == block.second.kernel_size == (3,)
                    and block.dropout.p == .25 for block in network.blocks))
                self.assertEqual(sum(isinstance(module, torch.nn.Dropout) for module in network.modules()), 5)
                values = torch.zeros((1, 130, 29))
                impulse = values.clone()
                impulse[:, 2, :] = 1.
                future = impulse.clone()
                future[:, 100:, :] = 100.
                with torch.no_grad():
                    response = network.hidden(impulse)
                    torch.testing.assert_close(response[:, :, :2], network.hidden(values)[:, :, :2], rtol=0, atol=0)
                    self.assertGreater(float(response[:, :, 126].min()), 0.)
                    torch.testing.assert_close(response[:, :, 127:], torch.zeros_like(response[:, :, 127:]), rtol=0, atol=0)
                    torch.testing.assert_close(response[:, :, :100], network.hidden(future)[:, :, :100], rtol=0, atol=0)
                    # Five complete supplied samples remain valid, despite RF125.
                    self.assertEqual(tuple(network(values[:, :5]).shape), (1, 19))

    def test_two_mean_original_and_factorial_slots_preserve_capacity_and_masked_invariance(self):
        with torch.random.fork_rng(devices=[]):
            original = backend.TCNCorrectionNetwork(16, recent_samples=2, adapter="original").eval()
            self.positive_weights(original)
            factorial = backend.TCNCorrectionNetwork(16, recent_samples=2, adapter="factorial").eval()
            factorial.load_state_dict(original.state_dict())
            x = torch.linspace(.1, 1., 290).reshape(1, 10, 29)
            changed_recent, changed_older = x.clone(), x.clone()
            changed_recent[:, 8:] += 100.
            changed_older[:, :8] += 10.
            with torch.no_grad():
                hidden, summary = original.hidden(x), original.summaries(x)
                torch.testing.assert_close(summary, torch.cat((hidden[:, :, :8].mean(2), hidden[:, :, 8:].mean(2)), 1))
                torch.testing.assert_close(summary[:, :16], original.summaries(changed_recent)[:, :16], rtol=0, atol=0)
                self.assertGreater(float((summary[:, 16:] - original.summaries(changed_older)[:, 16:]).abs().max()), 0.)
                torch.testing.assert_close(factorial.summaries(x)[:, 16:],
                                          factorial.summaries(changed_older)[:, 16:], rtol=0, atol=0)
                torch.testing.assert_close(factorial.summaries(x), torch.cat((factorial.hidden(x[:, :8]).mean(2),
                    factorial.hidden(x[:, 8:]).mean(2)), 1))
                blocks = FactorizedBlocks(x[0, :8].numpy(), np.ones((8, 29)) / 7., x[0, 8:].numpy())
                mutate_r = replace(blocks, recent=blocks.recent + 1000.)
                mutate_o = replace(blocks, older_slow=blocks.older_slow - 200.)
                def prediction(block, r, o):
                    return factorial(torch.tensor(block.reconstruct(r, o)[None], dtype=torch.float32))
                for o in (0, 1):
                    torch.testing.assert_close(prediction(blocks, 0, o), prediction(mutate_r, 0, o), rtol=0, atol=0)
                for r in (0, 1):
                    torch.testing.assert_close(prediction(blocks, r, 0), prediction(mutate_o, r, 0), rtol=0, atol=0)
                before = factorial(x)
                factorial(changed_recent)
                torch.testing.assert_close(before, factorial(x), rtol=0, atol=0)
            self.assertEqual([(name, tuple(value.shape)) for name, value in original.named_parameters()],
                             [(name, tuple(value.shape)) for name, value in factorial.named_parameters()])

    def test_registry_explicit_grid_and_complete_past_windows_gate_the_tcn(self):
        recipes = backend.tcn_recipes(REGISTRY)
        self.assertEqual(len(recipes), 12)
        self.assertEqual({recipe.family_parameters["channels"] for recipe in recipes}, {16, 32})
        modified = copy.deepcopy(REGISTRY)
        modified["reference_features"]["TCN_dilations"][-1] = 32
        with self.assertRaises(ValueError):
            backend.tcn_recipes(modified)
        def unopened():
            self.fail("invalid semantics must not consume fitting windows")
            yield []
        cases = [self.semantics(analysis_sample_rate_hz=7.), self.semantics(tcn_summary="last_hidden"),
                 self.semantics(tcn_boundary="cached"), self.semantics(channel_order=backend.CHANNELS[::-1]),
                 self.semantics(fitting_scope=FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9)))]
        absent = self.semantics()
        del absent["analysis_sample_rate_hz"]
        cases.append(absent)
        examples = self.examples()
        first = examples[0]
        windows = [replace(first.past_input, values=first.past_input.values[:-1]),
                   replace(first.past_input, sample_times=first.past_input.sample_times + .2),
                   replace(first.past_input, boundary_ids=["other", *first.past_input.boundary_ids[1:]])]
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            for semantics in cases:
                with self.subTest(semantics=semantics), self.assertRaises((ValueError, KeyError)):
                    backend.TCNForecastModel().fit(unopened(), recipe=recipes[0], seed=1901, semantics=semantics)
            for window in windows:
                with self.subTest(window=window), self.assertRaises(ValueError):
                    self.fit([replace(first, past_input=window), *examples[1:]])
            model = backend.TCNForecastModel()
            with self.assertRaisesRegex(UnresolvedFit, "no_observed_TCN"):
                self.fit([replace(item, observed_category=-1, complete_bins=0) for item in examples], model=model)
            self.assertFalse(model.fitted)
            forbidden.assert_not_called()

    def test_group_weights_offsets_and_masked_zeros_reach_training_unchanged(self):
        examples = self.examples("self_paced")
        zero = replace(examples[0], past_input=replace(examples[0].past_input, values=np.zeros((5, 29))))
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
        self.assertEqual(report["tcn_summary"], backend.TCN_SUMMARY)
        self.assertEqual(report["receptive_field_samples"], 125)
        self.assertFalse(report["source_admission_established_by_backend"])
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            with self.assertRaisesRegex(UnresolvedFit, "required fitting participant"):
                self.fit(examples[:-1], source="self_paced")
            forbidden.assert_not_called()

    def test_actual_fixed_schedule_budget_frozen_inference_and_zero_correction_offset(self):
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
        stopped = backend.TCNForecastModel()
        with self.assertRaisesRegex(RuntimeError, "allocation ended"):
            self.fit(model=stopped, budget_check=mock.Mock(side_effect=RuntimeError("allocation ended")))
        self.assertFalse(stopped.fitted)


if __name__ == "__main__":
    unittest.main()
