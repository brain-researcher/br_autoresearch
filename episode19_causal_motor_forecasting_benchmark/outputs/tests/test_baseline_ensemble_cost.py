"""Three focused cost-callback checks, initially UNRUN; coordinator Slurm only.

Existing backend objects are populated with explicit fitted-state algebra
fixtures. No fit method, training cohort, empirical input, model qualification,
or old test suite is executed. Prediction instrumentation exercises the existing
reset/predict interface while the actual fitted storage and reducer are used.
"""
from dataclasses import replace
from itertools import count
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from sklearn.tree import _tree

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import baseline_ensemble_cost as cost_module
from baseline_ensemble_cost import BaselineEnsembleCostMeasurer, BaselineInferenceProbe
from boosted_hazard_backend import BoostedHazardFit, BoostedHazardModel, boosted_hazard_recipes
from classical_forecast_backend import FittingScope, FittingStandardizer, LinearFit, UnresolvedFit
from ensemble_forecast_backend import SEEDS
from non_neural_forecast_backend import LogisticHazardModel, logistic_hazard_recipes
from temporal_hazard_backend import TemporalHazardModel, TemporalHazardNetwork, temporal_hazard_recipes
from test_development_baseline_driver import REGISTRY, row


SCHEMA = ("fixture_summary_a", "fixture_summary_b")
SEQUENCE_SCHEMA = ("fixture_sequence_a", "fixture_sequence_b")
SCOPE = FittingScope.development(REGISTRY, source="WAY",
    fold_id="WAY.P1.validation_series6", participant=1)


def probe(series=1, index=0):
    existing = row("WAY", 1, series, index)
    context = replace(existing.context, values=np.array([1., 2.]), feature_schema=SCHEMA)
    sequence = replace(existing.sequence, values=np.zeros((2, 2)), feature_schema=SEQUENCE_SCHEMA)
    return BaselineInferenceProbe(existing.identity, existing.decision_time, context, sequence)


def scaler():
    result = FittingStandardizer()
    result.mean, result.scale = np.zeros(2), np.ones(2)
    return result


def stored_tree(extra_split=False):
    """Native fitted tree storage assembled directly, without calling fit."""
    size = 5 if extra_split else 3
    nodes = np.zeros(size, dtype=_tree.NODE_DTYPE)
    nodes["left_child"] = [1, 3, -1, -1, -1] if extra_split else [1, -1, -1]
    nodes["right_child"] = [2, 4, -1, -1, -1] if extra_split else [2, -1, -1]
    nodes["feature"] = [0, 1, -2, -2, -2] if extra_split else [0, -2, -2]
    nodes["threshold"] = [.25, .75, -2., -2., -2.] if extra_split else [.25, -2., -2.]
    nodes["n_node_samples"] = 1
    nodes["weighted_n_node_samples"] = 1.
    tree = _tree.Tree(2, np.ones(1, dtype=np.intp), 1)
    tree.__setstate__(dict(max_depth=2 if extra_split else 1, node_count=size,
        nodes=nodes, values=np.arange(size, dtype=float).reshape(size, 1, 1) / 10.))
    return SimpleNamespace(tree_=tree)


def probabilities(seed):
    first = {1901: .1, 1902: .4, 1903: .8}[seed]
    return np.r_[first, np.full(18, (1. - first) / 18.)]


def instrument(model, tracker):
    original_reset = model.reset

    def reset(*, boundary_id):
        tracker.append(("reset", model.seed, boundary_id))
        original_reset(boundary_id=boundary_id)

    def predict(*inputs, decision_time):
        context = inputs[-1]
        if model.boundary_id != context.boundary_id or decision_time <= model.last_decision_time:
            raise AssertionError("cost replay failed to reset boundary/pass state")
        tracker.append(("predict", model.seed, decision_time))
        model.last_decision_time = decision_time
        return np.log(probabilities(model.seed))

    model.reset, model.predict_log_probabilities = reset, predict


def models(kind, tracker):
    factory, recipes = {
        "logistic": (LogisticHazardModel, logistic_hazard_recipes),
        "boosted": (BoostedHazardModel, boosted_hazard_recipes),
        "temporal": (TemporalHazardModel, temporal_hazard_recipes),
    }[kind]
    recipe, result = recipes(REGISTRY, source="WAY")[0], []
    for seed in SEEDS:
        model = factory()
        model.recipe, model.seed, model.fitting_scope, model.fitted = recipe, seed, SCOPE, True
        model.boundary_id, model.last_decision_time = None, -np.inf
        if kind == "temporal":
            model.sequence_schema, model.summary_schema = SEQUENCE_SCHEMA, SCHEMA
            model.analysis_sample_rate_hz = 1.
            model.sequence_standardizer, model.summary_standardizer = scaler(), scaler()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                model.network = TemporalHazardNetwork(2, 2).eval()
            for parameter in model.network.parameters():
                parameter.requires_grad_(False)
        else:
            model.feature_schema, model.standardizer = SCHEMA, scaler()
            if kind == "logistic":
                model.fit_result = LinearFit(np.zeros((2, 18)), np.zeros(18), True,
                    0, 0., 0., "explicit stored-state fixture; no fitting")
            else:
                heads = []
                for head_index in range(18):
                    estimators = np.empty((200, 1), dtype=object)
                    for tree_index in range(200):
                        estimators[tree_index, 0] = stored_tree(
                            seed == 1901 and head_index == 0 and tree_index == 0)
                    heads.append(SimpleNamespace(init="zero", learning_rate=recipe.learning_rate,
                        n_estimators_=200, estimators_=estimators))
                model.fit_result = BoostedHazardFit(tuple(heads), 2, True, 0., 0.,
                    "explicit stored-tree fixture; no fitting")
        instrument(model, tracker)
        result.append(model)
    return tuple(result), recipe


def measurer(probes, *, warmup=0, passes=1, check=lambda: None):
    return BaselineEnsembleCostMeasurer(registry=REGISTRY, probes_by_scope={SCOPE: probes},
        analysis_sample_rate_hz=1., warmup_passes=warmup, measurement_passes=passes, budget_check=check)


class BaselineEnsembleCostTests(unittest.TestCase):
    def test_actual_fitted_storage_across_all_backends_includes_scalers_and_frozen_parameters(self):
        # Logistic:2*18 coefficients+18 intercepts+2 means+2 scales.
        # Trees:18*200*(2 coordinates per split+2 leaves), plus4 scaler
        # values; one actual five-node tree adds one split and one leaf.
        # Temporal: first residual block944, four later blocks4*1568,
        # readout18*(16+2)+18, and two mean/scale pairs of size2.
        expected = {"logistic": 3 * 58, "boosted": 3 * 14404 + 3,
                    "temporal": 3 * (944 + 4 * 1568 + 342 + 8)}
        for kind, total in expected.items():
            events = []
            fitted, recipe = models(kind, events)
            measured = measurer((probe(),))
            with self.subTest(backend=kind), mock.patch.object(cost_module, "perf_counter_ns",
                    side_effect=(0, 1000000)):
                result = measured(models=fitted, recipe=recipe, fitting_scope=SCOPE)
            self.assertEqual(result.parameter_count, total)
            components = measured.measurements[-1]["per_seed_components"]
            if kind == "boosted":
                self.assertEqual(components[0]["split_feature_choices"], 3601)
                self.assertEqual(components[0]["leaf_scores"], 7201)
                self.assertEqual(components[1]["split_thresholds"], 3600)
            if kind == "temporal":
                self.assertEqual(components[0]["network_parameters"], 7558)
                self.assertTrue(all(not parameter.requires_grad for model in fitted
                                    for parameter in model.network.parameters()))
            self.assertEqual(len([event for event in events if event[0] == "predict"]), 3)

    def test_probability_mean_is_one_timed_decision_with_resets_between_every_pass_and_boundary(self):
        events, means = [], []
        fitted, recipe = models("logistic", events)
        panel = (probe(1, 0), probe(1, 1), probe(2, 0))
        measured = measurer(panel, warmup=1, passes=2, check=lambda: events.append(("check",)))
        durations = tuple(index * 1000000 for index in range(1, 7))
        clock_index = count()

        def clock():
            index = next(clock_index)
            events.append(("clock", index))
            return (index // 2) * 10000000 + (durations[index // 2] if index % 2 else 0)

        original_mean = cost_module.mean_seed_log_probabilities

        def mean(*args, **kwargs):
            value = original_mean(*args, **kwargs)
            means.append(value.copy())
            events.append(("mean",))
            return value

        with mock.patch.object(cost_module, "perf_counter_ns", side_effect=clock) as timer, \
             mock.patch.object(cost_module, "mean_seed_log_probabilities", side_effect=mean):
            result = measured(models=fitted, recipe=recipe, fitting_scope=SCOPE)
        self.assertEqual(timer.call_count, 12)  # Six complete decisions, never18 seed timers.
        self.assertEqual(len(means), 9)  # Three warmup decisions plus six measured.
        expected = sum(probabilities(seed) for seed in SEEDS) / 3.
        for value in means:
            np.testing.assert_allclose(np.exp(value[0]), expected, rtol=0., atol=1e-14)
        self.assertEqual(result.latency_seconds, .006)  # nearest-rank p95 of1..6ms
        self.assertEqual(measured.measurements[-1]["latency_nanoseconds"], durations)
        clock_events = [index for index, event in enumerate(events) if event[0] == "clock"]
        for start, end in zip(clock_events[::2], clock_events[1::2]):
            self.assertEqual([event[0] for event in events[start + 1:end]],
                             ["predict", "predict", "predict", "mean"])
            self.assertEqual([event[1] for event in events[start + 1:end - 1]], list(SEEDS))
        for model in fitted:
            self.assertEqual(len([event for event in events if event[:2] == ("reset", model.seed)]), 7)
            self.assertEqual(model.boundary_id, panel[0].identity.boundary_id)
            self.assertEqual(model.last_decision_time, -np.inf)

    def test_validation_identities_and_wrong_seeds_refuse_before_inference_and_budget_failure_cleans_up(self):
        events = []
        fitted, recipe = models("logistic", events)
        with self.assertRaisesRegex(ValueError, "fitting fold"):
            measurer((probe(6),))  # canonical scope fits series1..5; series6 validates
        self.assertFalse(any(event[0] == "predict" for event in events))
        measured = measurer((probe(),))
        fitted[-1].seed = 999
        with mock.patch.object(cost_module, "perf_counter_ns") as clock:
            with self.assertRaises(UnresolvedFit):
                measured(models=fitted, recipe=recipe, fitting_scope=SCOPE)
            clock.assert_not_called()
        self.assertFalse(any(event[0] == "predict" for event in events))
        self.assertEqual(measured.measurements, [])
        fitted[-1].seed = SEEDS[-1]
        stopped = RuntimeError("fixture allocation ended")

        def check():
            if any(event[0] == "predict" for event in events):
                raise stopped

        measured = measurer((probe(1, 0), probe(1, 1)), check=check)
        with mock.patch.object(cost_module, "perf_counter_ns", side_effect=(0, 1000000)):
            with self.assertRaises(RuntimeError) as error:
                measured(models=fitted, recipe=recipe, fitting_scope=SCOPE)
        self.assertIs(error.exception, stopped)
        self.assertEqual(len([event for event in events if event[0] == "predict"]), 3)
        self.assertEqual(measured.measurements, [])
        for model in fitted:
            self.assertEqual(model.boundary_id, probe().identity.boundary_id)
            self.assertEqual(model.last_decision_time, -np.inf)


if __name__ == "__main__":
    unittest.main()
