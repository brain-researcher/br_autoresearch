"""Six focused array-only GRU checks; coordinator Slurm execution only.

Includes one tiny actual100-epoch fit, not a synthetic scientific study,
source/timing qualification or empirical forecast. No protected data are read.
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
import gru_forecast_backend as backend
from classical_forecast_backend import (
    FittingExample, FittingScope, GroupIdentity, PastWindow, UnresolvedFit,
    normalized_log_probabilities,
)
from forecast_components import FactorizedBlocks, censoring_loss_and_logit_gradient

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


class GRUBackendTests(unittest.TestCase):
    def scope(self, source="WAY"):
        return FittingScope.development(REGISTRY, source=source,
            fold_id="WAY.P1.validation_series6" if source == "WAY" else "self_paced.f0",
            participant=1 if source == "WAY" else None)

    def example(self, *, person=1, group=1, source="WAY", shift=0., category=18):
        decision, rate, count = 10., 5., 5
        boundary = f"{source}-{person}-{group}"
        values = (np.arange(count * 29).reshape(count, 29) / 100. + shift)
        times = decision - np.arange(count, 0, -1) / rate
        offset = normalized_log_probabilities(np.linspace(-2., 1., 19)[None, :])[0]
        return FittingExample(PastWindow(values, times, [boundary] * count), decision,
            GroupIdentity(source, person, group, boundary), category, 18, offset)

    def examples(self, source="WAY"):
        scope = self.scope(source)
        groups = scope.allowed_series if source == "WAY" else scope.allowed_participants
        return [self.example(person=1 if source == "WAY" else group,
                    group=group if source == "WAY" else "run-1", source=source, shift=index / 10.)
                for index, group in enumerate(groups)]

    def semantics(self, source="WAY", **changes):
        result = dict(registry=REGISTRY, fitting_scope=self.scope(source), analysis_sample_rate_hz=5.,
            channel_order=backend.CHANNELS, adapter="original", budget_check=lambda: None)
        result.update(changes)
        return result

    def fit(self, examples=None, *, model=None, source="WAY", seed=1901, **semantics):
        model = model or backend.GRUForecastModel()
        report = model.fit([self.examples(source) if examples is None else examples],
            recipe=backend.gru_recipes(REGISTRY)[0], seed=seed, semantics=self.semantics(source, **semantics))
        return model, report

    def test_censored_likelihood_gradient_offsets_and_global_weights(self):
        correction = np.linspace(-3., 3., 114).reshape(6, 19)
        offsets = normalized_log_probabilities(np.sin(correction))
        category, complete = np.array([0, 8, 18, -1, -1, -1]), np.array([0, 9, 18, 7, 0, 18])
        logits = correction + offsets
        loss, gradient = censoring_loss_and_logit_gradient(normalized_log_probabilities(logits), category, complete)
        tensor = torch.tensor(correction, dtype=torch.float64, requires_grad=True)
        labels, bins = torch.tensor(category), torch.tensor(complete)
        fixed = torch.tensor(offsets, dtype=torch.float64)
        actual = backend.censored_category_losses(tensor + fixed, labels, bins)
        np.testing.assert_allclose(actual.detach().numpy(), loss, atol=1e-12)
        actual.sum().backward()
        np.testing.assert_allclose(tensor.grad.numpy(), gradient, atol=1e-12)
        self.assertEqual(float(actual[4]), 0.)
        self.assertIsNone(fixed.grad)
        np.testing.assert_array_equal(fixed.numpy(), offsets)
        weights = torch.tensor([.05, .05, .1, .2, .2, .4], dtype=torch.float64)
        first = backend._minibatch_objective(tensor[:3], fixed[:3], labels[:3], bins[:3], weights[:3], 6)
        second = backend._minibatch_objective(tensor[3:], fixed[3:], labels[3:], bins[3:], weights[3:], 6)
        self.assertAlmostEqual(float((first + second) / 2), float(weights @ actual), places=12)
        self.assertAlmostEqual(float(first), 2 * float(weights[:3] @ actual[:3]), places=12)

    def test_causal_recurrence_shared_capacity_and_masked_block_invariance(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1901)
            original = backend.GRUCorrectionNetwork(16, recent_samples=2, adapter="original").eval()
            factorial = backend.GRUCorrectionNetwork(16, recent_samples=2, adapter="factorial").eval()
            factorial.load_state_dict(original.state_dict())
            x = torch.linspace(-1., 1., 290).reshape(1, 10, 29)
            changed_recent, changed_older = x.clone(), x.clone()
            changed_recent[:, 8:] += 100.
            changed_older[:, :8] -= 10.
            with torch.no_grad():
                continuous, _ = original.encoder(x)
                summary = original.summaries(x)
                torch.testing.assert_close(summary[:, :16], continuous[:, 7], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(summary[:, 16:], continuous[:, -1], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(summary[:, :16], original.summaries(changed_recent)[:, :16], rtol=0, atol=0)
                torch.testing.assert_close(factorial.summaries(x)[:, 16:],
                                          factorial.summaries(changed_older)[:, 16:], rtol=0, atol=0)
                self.assertGreater(float(torch.max(torch.abs(summary[:, 16:] - original.summaries(changed_older)[:, 16:]))), 0.)
                blocks = FactorizedBlocks(x[0, :8].numpy(), np.ones((8, 29)) / 7., x[0, 8:].numpy())
                mutate_r = replace(blocks, recent=blocks.recent + 1000.)
                mutate_o = replace(blocks, older_slow=blocks.older_slow - 200.)
                def prediction(block, r, o):
                    return factorial(torch.tensor(block.reconstruct(r, o)[None], dtype=torch.float32))
                for o in (0, 1):
                    torch.testing.assert_close(prediction(blocks, 0, o), prediction(mutate_r, 0, o), rtol=0, atol=0)
                for r in (0, 1):
                    torch.testing.assert_close(prediction(blocks, r, 0), prediction(mutate_o, r, 0), rtol=0, atol=0)
            self.assertFalse(original.encoder.bidirectional)
            self.assertEqual(original.encoder.num_layers, 1)
            self.assertEqual(original.dropout.p, .25)
            self.assertEqual(original.readout.in_features, 32)
            self.assertEqual([p.shape for p in original.parameters()], [p.shape for p in factorial.parameters()])

    def test_recipe_scope_and_unresolved_semantics_rejected_before_batches(self):
        recipes = backend.gru_recipes(REGISTRY)
        self.assertEqual(len(recipes), 12)
        self.assertEqual({recipe.family_parameters["hidden_units"] for recipe in recipes}, {16, 32})
        def unopened():
            self.fail("invalid scope or unspecified recipe must not consume batches")
            yield []
        forged = FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9))
        cases = [self.semantics(fitting_scope=forged), self.semantics(channel_order=backend.CHANNELS[::-1]),
                 self.semantics(analysis_sample_rate_hz=7.), self.semantics(gru_summary="mean"),
                 self.semantics(gru_initial_state="cached"), self.semantics(adapter="unselected")]
        absent = self.semantics()
        del absent["analysis_sample_rate_hz"]
        cases.append(absent)
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            for semantics in cases:
                with self.subTest(semantics=semantics), self.assertRaises((ValueError, KeyError)):
                    backend.GRUForecastModel().fit(unopened(), recipe=recipes[0], seed=1901, semantics=semantics)
            with self.assertRaises(ValueError):
                self.fit(seed=1904)
            changed = copy.deepcopy(REGISTRY)
            changed["training"]["neural_epochs"] = 99
            with self.assertRaises(ValueError):
                self.fit(registry=changed)
            forbidden.assert_not_called()

    def test_group_weights_keep_shared_scaled_zero_masks_and_fixed_offsets(self):
        examples = self.examples("self_paced")
        zero = replace(examples[0], past_input=replace(examples[0].past_input,
                                                     values=np.zeros((5, 29))))
        examples = [zero, examples[0], *examples[1:]]
        supplied = np.asarray([example.past_input.values for example in examples]).copy()
        offsets = np.asarray([example.context_log_probabilities for example in examples]).copy()
        with mock.patch.object(backend, "_train_fixed_schedule", return_value=(object(), 100, 1.)) as train:
            _, report = self.fit(examples, source="self_paced", adapter="factorial")
        np.testing.assert_array_equal(train.call_args.args[0], supplied)
        np.testing.assert_array_equal(train.call_args.args[0][0], 0.)
        np.testing.assert_array_equal(train.call_args.args[1], offsets)
        weights = train.call_args.args[4]
        np.testing.assert_allclose(weights[:2], 1 / 24)
        np.testing.assert_allclose(weights[2:], 1 / 12)
        self.assertEqual(train.call_args.kwargs["adapter"], "factorial")
        self.assertFalse(report["source_admission_established_by_backend"])
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            with self.assertRaisesRegex(UnresolvedFit, "required fitting participant"):
                self.fit(examples[:-1], source="self_paced")
            with self.assertRaisesRegex(UnresolvedFit, "required fitting series"):
                self.fit(self.examples()[:-1])
            forbidden.assert_not_called()

    def test_incomplete_future_complex_and_censored_inputs_never_train(self):
        examples = self.examples()
        first = examples[0]
        invalid_windows = [replace(first.past_input, values=first.past_input.values[:-1]),
            replace(first.past_input, values=np.full((5, 29), np.nan)),
            replace(first.past_input, values=first.past_input.values.astype(complex) + 1j),
            replace(first.past_input, sample_times=first.past_input.sample_times + .2),
            replace(first.past_input, boundary_ids=["other", *first.past_input.boundary_ids[1:]])]
        model = backend.GRUForecastModel()
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            for window in invalid_windows:
                with self.subTest(window=window), self.assertRaises(ValueError):
                    self.fit([replace(first, past_input=window), *examples[1:]], model=model)
            with self.assertRaisesRegex(ValueError, "normalized"):
                self.fit([replace(first, context_log_probabilities=np.zeros(19)), *examples[1:]])
            with self.assertRaisesRegex(UnresolvedFit, "no_observed_GRU"):
                self.fit([replace(item, observed_category=-1, complete_bins=0) for item in examples], model=model)
            with self.assertRaisesRegex(RuntimeError, "allocation ended"):
                self.fit(model=model, budget_check=mock.Mock(side_effect=RuntimeError("allocation ended")))
            forbidden.assert_not_called()
        self.assertFalse(model.fitted)
        model.reset(boundary_id=first.identity.boundary_id)
        with self.assertRaises(UnresolvedFit):
            model.predict_log_probabilities(first.past_input, decision_time=10., context_log_probabilities=first.context_log_probabilities)

    def test_actual_100_epoch_schedule_frozen_inference_and_rng_restoration(self):
        self.assertEqual(torch.get_num_threads(), 1, "coordinator must set one numerical thread")
        events, defaults = [], []
        original_step = torch.optim.AdamW.step
        def step(optimizer, *args, **kwargs):
            self.assertEqual(events[-1], "check")
            events.append("step")
            if not defaults:
                defaults.append(optimizer.defaults.copy())
            return original_step(optimizer, *args, **kwargs)
        rng = torch.random.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        examples = self.examples()
        examples[0] = replace(examples[0], observed_category=0, complete_bins=0)
        examples[1] = replace(examples[1], observed_category=-1, complete_bins=9)
        offsets_before = np.asarray([item.context_log_probabilities for item in examples]).copy()
        with mock.patch.object(torch.optim.AdamW, "step", new=step):
            model, report = self.fit(examples, budget_check=lambda: events.append("check"))
        self.assertTrue(report["completed"])
        self.assertEqual(report["epochs"], 100)
        self.assertEqual(report["optimizer_steps"], 100)
        self.assertEqual(events.count("step"), 100)
        self.assertTrue(np.isfinite(report["objective"]))
        for key, value in dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.001,
                amsgrad=False, maximize=False, foreach=False, capturable=False,
                differentiable=False, fused=False).items():
            self.assertEqual(defaults[0][key], value)
        self.assertFalse(model.network.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None and p.dtype == torch.float32
                            and p.device.type == "cpu" for p in model.network.parameters()))
        np.testing.assert_array_equal(offsets_before, np.asarray([item.context_log_probabilities for item in examples]))
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn)
        example = self.example(group=7, shift=100.)
        model.reset(boundary_id=example.identity.boundary_id)
        logp = model.predict_log_probabilities(example.past_input, decision_time=10.,
                                               context_log_probabilities=example.context_log_probabilities)
        self.assertEqual(logp.shape, (19,))
        self.assertAlmostEqual(float(np.exp(logp).sum()), 1., places=12)
        with self.assertRaisesRegex(ValueError, "increasing"):
            model.predict_log_probabilities(example.past_input, decision_time=10.,
                                            context_log_probabilities=example.context_log_probabilities)
        model.reset(boundary_id=example.identity.boundary_id)
        np.testing.assert_array_equal(logp, model.predict_log_probabilities(example.past_input,
            decision_time=10., context_log_probabilities=example.context_log_probabilities))
        # Zero correction must recover the *supplied* non-neural distribution;
        # there is no separately refit baseline, probability floor or offset bias.
        with torch.no_grad():
            model.network.readout.weight.zero_()
            model.network.readout.bias.zero_()
        model.reset(boundary_id=example.identity.boundary_id)
        actual = model.predict_log_probabilities(example.past_input, decision_time=10.,
                                                 context_log_probabilities=example.context_log_probabilities)
        np.testing.assert_allclose(actual, example.context_log_probabilities, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
