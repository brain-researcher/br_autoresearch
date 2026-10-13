"""Five source-free checks; run only in the coordinator's Slurm allocation.

One tiny five-example fit completes the actual fixed 100-epoch schedule.
No source timing, sensor calibration, empirical readiness or fit is tested.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import temporal_hazard_backend as backend
from classical_forecast_backend import FittingScope, GroupIdentity, UnresolvedFit, hazard_loss_and_gradient
from non_neural_forecast_backend import ContextFeatures

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())
SEQUENCE_SCHEMA = ("prepared_channel_a", "prepared_channel_b")
SUMMARY_SCHEMA = ("prepared_summary",)


class TemporalHazardTests(unittest.TestCase):
    def scope(self, source="WAY"):
        return FittingScope.development(REGISTRY, source=source,
            fold_id="WAY.P1.validation_series6" if source == "WAY" else "self_paced.f0",
            participant=1 if source == "WAY" else None)

    def example(self, *, person=1, group=1, source="WAY", rate=2., offset=0.):
        count, decision = int(2 * rate), 10.
        times = decision - np.arange(count, 0, -1) / rate
        boundary = f"{source}-{person}-{group}"
        values = np.column_stack((np.arange(count) + offset, np.full(count, 3.)))
        sequence = backend.ContextSequence(values, SEQUENCE_SCHEMA, source, person,
            (boundary,) * count, times, times.copy())
        summary = ContextFeatures([offset + 1.], SUMMARY_SCHEMA, source, person, boundary, decision)
        return backend.TemporalFittingExample(sequence, summary, decision,
            GroupIdentity(source, person, group, boundary), 18, 18)

    def examples(self, source="WAY"):
        scope = self.scope(source)
        groups = scope.allowed_series if source == "WAY" else scope.allowed_participants
        return [self.example(person=1 if source == "WAY" else group,
                    group=group if source == "WAY" else "run-1", source=source, offset=index)
                for index, group in enumerate(groups)]

    def fit(self, examples=None, *, model=None, source="WAY", **changes):
        arguments = dict(recipe=backend.temporal_hazard_recipes(REGISTRY, source=source)[0],
            seed=1901, registry=REGISTRY, fitting_scope=self.scope(source),
            sequence_schema=SEQUENCE_SCHEMA, summary_schema=SUMMARY_SCHEMA,
            analysis_sample_rate_hz=2., budget_check=lambda: None)
        arguments.update(changes)
        model = model or backend.TemporalHazardModel()
        report = model.fit([self.examples(source) if examples is None else examples], **arguments)
        return model, report

    def test_censored_loss_gradient_and_global_minibatch_weights(self):
        raw = np.linspace(-3., 4., 72).reshape(4, 18)
        category, complete = [0, 7, 18, -1], [18, 18, 18, 0]
        risk, events = backend.conditional_hazard_targets(category, complete, 4)
        logits = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        risk_t, events_t = torch.tensor(risk), torch.tensor(events)
        expected_loss, expected_gradient = hazard_loss_and_gradient(raw, category, complete)
        losses = backend.censored_hazard_losses(logits, risk_t, events_t)
        np.testing.assert_allclose(losses.detach().numpy(), expected_loss, atol=1e-12)
        losses.sum().backward()
        np.testing.assert_allclose(logits.grad.numpy(), expected_gradient, atol=1e-12)
        self.assertEqual(float(losses[-1]), 0.)
        weights = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
        first = backend._minibatch_objective(logits[:2], risk_t[:2], events_t[:2], weights[:2], 4)
        second = backend._minibatch_objective(logits[2:], risk_t[2:], events_t[2:], weights[2:], 4)
        self.assertAlmostEqual(float((first + second) / 2), float(weights @ losses), places=12)
        # The first batch total .3 is deliberately not normalized to one.
        self.assertAlmostEqual(float(first), 2 * float(weights[:2] @ losses[:2]), places=12)

    def test_complete_40_sample_grid_and_causal_finite_window_operator(self):
        example = self.example(rate=20.)
        arguments = dict(schema=SEQUENCE_SCHEMA, source="WAY", participant=1,
            boundary_id=example.identity.boundary_id, decision_time=10., sample_rate_hz=20.)
        values = example.sequence.validated(**arguments)
        self.assertEqual(values.shape, (40, 2))  # No receptive-field minimum of 125.
        invalid = [replace(example.sequence, values=values[:-1]),
            replace(example.sequence, values=np.full_like(values, np.nan)),
            replace(example.sequence, values=values.astype(complex) + 1j),
            replace(example.sequence, sample_times=np.roll(example.sequence.sample_times, 1)),
            replace(example.sequence, sample_times=example.sequence.sample_times + .05),
            replace(example.sequence, available_through=example.sequence.sample_times + .001),
            replace(example.sequence, boundary_ids=("other",) + example.sequence.boundary_ids[1:])]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                item.validated(**arguments)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1901)
            network = backend.TemporalHazardNetwork(2, 1).eval()
            sequence = torch.tensor(values.T[None], dtype=torch.float32)
            changed = sequence.clone()
            changed[:, :, 20:] += 1000.
            with torch.no_grad():
                hidden = network.hidden(sequence)
                later_changed = network.hidden(changed)
                repeated = network.hidden(sequence)
                output = network(sequence, torch.zeros((1, 1)))
            self.assertEqual(tuple(hidden.shape), (1, 16, 40))
            torch.testing.assert_close(hidden[:, :, :20], later_changed[:, :, :20], rtol=0, atol=0)
            torch.testing.assert_close(hidden, repeated, rtol=0, atol=0)
            self.assertEqual(tuple(output.shape), (1, 18))
            self.assertEqual([block.left for block in network.blocks], [2, 4, 8, 16, 32])

    def test_group_roles_and_training_only_channel_summary_scaling(self):
        examples = self.examples("self_paced")
        examples = [examples[0], replace(examples[0], sequence=replace(examples[0].sequence,
                     values=examples[0].sequence.values + 2.)), *examples[1:]]
        with mock.patch.object(backend, "_train_fixed_schedule", return_value=(object(), 100, 1.)) as fit:
            model, report = self.fit(examples, source="self_paced")
        weights = fit.call_args.args[4]
        np.testing.assert_allclose(weights[:2], 1/24)
        np.testing.assert_allclose(weights[2:], 1/12)
        raw = np.asarray([item.sequence.values for item in examples])
        expected = np.sum(raw.mean(axis=1) * weights[:, None], axis=0)
        np.testing.assert_allclose(model.sequence_standardizer.mean, expected)
        np.testing.assert_allclose(model.summary_standardizer.mean,
                                   np.sum([item.summary.values[0] * weight for item, weight in zip(examples, weights)]))
        np.testing.assert_allclose(np.sum(fit.call_args.args[0].mean(axis=1) * weights[:, None], axis=0), 0., atol=1e-12)
        self.assertFalse(report["source_admission_established_by_backend"])
        def unopened():
            self.fail("forged role must be rejected before reading caller batches")
            yield []
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            with self.assertRaisesRegex(ValueError, "canonical development fold"):
                backend.TemporalHazardModel().fit(unopened(),
                    recipe=backend.temporal_hazard_recipes(REGISTRY, source="WAY")[0], seed=1901,
                    registry=REGISTRY, fitting_scope=FittingScope("WAY", "WAY.P1.validation_series6", (1,), (8, 9)),
                    sequence_schema=SEQUENCE_SCHEMA, summary_schema=SUMMARY_SCHEMA,
                    analysis_sample_rate_hz=2., budget_check=lambda: None)
            with self.assertRaisesRegex(UnresolvedFit, "required fitting series"):
                self.fit(self.examples()[:-1])
            forbidden.assert_not_called()

    def test_unobserved_heads_and_budget_stop_before_any_fit(self):
        unobserved = [replace(item, observed_category=-1, complete_bins=17) for item in self.examples()]
        model = backend.TemporalHazardModel()
        with mock.patch.object(backend, "_train_fixed_schedule") as forbidden:
            with self.assertRaisesRegex(UnresolvedFit, "no_observed_conditional_intervals_heads_\\[17\\]"):
                self.fit(unobserved, model=model)
            with self.assertRaisesRegex(RuntimeError, "allocation ended"):
                self.fit(model=model, budget_check=mock.Mock(side_effect=RuntimeError("allocation ended")))
            with self.assertRaisesRegex(ValueError, "registered temporal recipe"):
                self.fit(seed=999)
            forbidden.assert_not_called()
        self.assertFalse(model.fitted)
        example = self.example()
        model.reset(boundary_id=example.identity.boundary_id)
        with self.assertRaises(UnresolvedFit):
            model.predict_log_probabilities(example.sequence, example.summary, decision_time=10.)

    def test_one_tiny_actual_100_epoch_fit_single_class_heads_and_frozen_inference(self):
        self.assertEqual(torch.get_num_threads(), 1, "coordinator must limit this test allocation to one thread")
        events = []
        optimizer_defaults = []
        original_step = torch.optim.AdamW.step
        def step(optimizer, *args, **kwargs):
            self.assertEqual(events[-1], "check")
            events.append("step")
            if not optimizer_defaults:
                optimizer_defaults.append(optimizer.defaults.copy())
            return original_step(optimizer, *args, **kwargs)
        rng = torch.random.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        adamw = torch.optim.AdamW
        with mock.patch.object(adamw, "step", new=step):
            model, report = self.fit(budget_check=lambda: events.append("check"))
        self.assertTrue(report["completed"])
        self.assertEqual(report["epochs"], 100)
        self.assertEqual(report["optimizer_steps"], 100)
        self.assertEqual(events.count("step"), 100)
        self.assertEqual(report["single_class_observed_heads"], list(range(18)))
        self.assertTrue(np.isfinite(report["objective"]))
        for key, value in dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.001,
                amsgrad=False, maximize=False, foreach=False, capturable=False,
                differentiable=False, fused=False).items():
            self.assertEqual(optimizer_defaults[0][key], value)
        self.assertTrue(all(not p.requires_grad and p.grad is None and p.dtype == torch.float32
                            and p.device.type == "cpu" for p in model.network.parameters()))
        self.assertFalse(model.network.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn_only)
        example = self.example(group=7, offset=100.)
        means = model.sequence_standardizer.mean.copy(), model.summary_standardizer.mean.copy()
        model.reset(boundary_id=example.identity.boundary_id)
        with self.assertRaisesRegex(ValueError, "participant"):
            model.predict_log_probabilities(replace(example.sequence, participant=2), example.summary, decision_time=10.)
        self.assertEqual(model.last_decision_time, -np.inf)
        logp = model.predict_log_probabilities(example.sequence, example.summary, decision_time=10.)
        self.assertEqual(logp.shape, (19,))
        self.assertTrue(np.isfinite(logp).all())
        self.assertAlmostEqual(float(np.exp(logp).sum()), 1., places=12)
        with self.assertRaisesRegex(ValueError, "increasing"):
            model.predict_log_probabilities(example.sequence, example.summary, decision_time=10.)
        model.reset(boundary_id=example.identity.boundary_id)
        np.testing.assert_array_equal(logp, model.predict_log_probabilities(example.sequence, example.summary, decision_time=10.))
        np.testing.assert_array_equal(means[0], model.sequence_standardizer.mean)
        np.testing.assert_array_equal(means[1], model.summary_standardizer.mean)


if __name__ == "__main__":
    unittest.main()
