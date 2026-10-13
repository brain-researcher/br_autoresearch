"""Focused orchestration checks; invented arrays, coordinated compute only.

Existing kernels/selector/refit have their own passing numerical tests. Here
instrumented kernels let the real complete PCA/model loops exercise ordering,
coverage, shared inputs, canonical assembly and failure without another costly
numerical benchmark. The overlap check requires >=2 allocated CPUs; other
checks use one worker and retain the complete four-model scope.
"""
from collections import Counter
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from threading import Barrier, Lock
import time
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import parallel_nested_prediction as parallel
import nested_prediction as nested


def inputs():
    concepts = tuple(i for i in range(480) for _ in range(6 if i < 23 else 5))
    design = nested.DevelopmentDesign(tuple(f"image_{i}" for i in range(2423)),
        tuple(f"concept_{i}" for i in concepts), ("development",) * 2423,
        tuple(i // 80 for i in concepts), tuple(i % 24 for i in concepts))
    # Already-bound input doubles avoid repeating the previously tested 300 MB
    # feature snapshot. Every scientific row/column label remains explicit.
    values = nested.endpoint.frozen_array(np.arange(2423, dtype=float)[:, None])
    features = {model: {layer: values for layer in nested.LAYERS} for model in nested.MODEL_IDS}
    responses = {group: {stage: values for stage in nested.STAGES} for group in nested.GROUPS}
    columns = {group: nested.endpoint.column_ids(np.array([7]), "fixture") for group in nested.GROUPS}
    return design, features, responses, columns


class ParallelPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import model_features
        model_features.check_runtime()

    def _instrument(self, stack, source, events, *, fail_pca=False, fail_model=False,
                    require_shared_mapping=True):
        design, features, responses, columns = source
        calls = Counter()
        stack.enter_context(patch.object(nested, "_bind_inputs", return_value=(features, responses, columns)))

        def pca(values, rank):
            calls["pca"] += 1
            if fail_pca and calls["pca"] == 19:
                raise ValueError("invented numerical refusal")
            self.assertEqual(rank, 128)
            return (len(values), rank)

        stack.enter_context(patch.object(nested.core, "fit_pca", new=pca))
        stack.enter_context(patch.object(nested, "_frozen_pca", new=lambda value: value))
        stack.enter_context(patch.object(nested, "derive_rank_pca", new=lambda fit, values, rank: (fit[0], rank)))

        def prepare(design, observed, training, validation, blocks, check):
            calls["prepare"] += 1
            self.assertEqual(calls["pca"], 352)
            self.assertIn(("all_features_feasible", "after"), events)
            self.assertEqual(len(training), 22)
            self.assertEqual(len(validation), 6)
            self.assertEqual(len(blocks), 24)
            means = {folds: {group: {stage: np.array([np.mean(observed[group][stage][rows])])
                for stage in nested.STAGES} for group in nested.GROUPS} for folds, rows in training.items()}
            baselines = {group: {stage: nested.endpoint.frozen_array(observed[group][stage])
                for stage in nested.STAGES} for group in nested.GROUPS}
            return means, baselines

        def select(model, folds, feats, observed, training, validation, pcas, means, check):
            calls["select"] += 1
            self.assertEqual(calls["prepare"], 1)
            self.assertEqual(set(observed), set(nested.GROUPS))
            self.assertTrue(all(set(value) == set(nested.STAGES) for value in observed.values()))
            self.assertEqual(len(pcas), 352)
            self.assertTrue(all(set(value) == {32, 64, 128} for value in pcas.values()))
            self.assertEqual(len(nested.SETTINGS), 84)
            # Only the parallel orchestrator adds read-only mapping wrappers.
            # The original serial reference still supplies immutable arrays.
            if require_shared_mapping:
                self.assertIsInstance(feats, MappingProxyType)
            self.assertFalse(feats[model]["layer1"].flags.writeable)
            self.assertEqual(len(folds), 5 if calls["select"] % 7 else 6)
            if fail_model and model == nested.MODEL_IDS[1]:
                raise ValueError("invented selector refusal")
            objectives = MappingProxyType({setting: 1. for setting in nested.SETTINGS})
            winner = min(nested.SETTINGS, key=lambda setting:
                nested.core.hyperparameter_tie_key(setting.layer, setting.rank, setting.penalty))
            return nested.Selection(winner, objectives)

        def refit(model, folds, selection, feats, observed, training, pcas, means, check,
                  *, test_rows=None, destination=None):
            calls["refit"] += 1
            token = float(nested.MODEL_IDS.index(model) * 100 + sum(folds))
            if test_rows is not None:
                self.assertEqual(len(folds), 5)
                for group in nested.GROUPS:
                    for stage in nested.STAGES:
                        destination[group][stage][test_rows] = token
            else:
                self.assertEqual(len(folds), 6)
            return (len(training[folds]), selection.setting.rank), (len(training[folds]), 64), MappingProxyType({})

        stack.enter_context(patch.object(nested, "_prepare_responses", new=prepare))
        stack.enter_context(patch.object(nested, "_select", new=select))
        stack.enter_context(patch.object(nested, "_refit", new=refit))
        return calls

    def test_complete_panel_matches_serial_orchestration_and_352_barrier(self):
        source, events = inputs(), []
        with ExitStack() as stack:
            calls = self._instrument(stack, source, events, require_shared_mapping=False)
            check = lambda event: events.append((event.operation, event.when))
            serial = nested.run_nested_prediction(*source, resource_check=check)
            self.assertEqual(calls, Counter(pca=352, prepare=1, select=28, refit=28))
        events.clear()
        with ExitStack() as stack:
            calls = self._instrument(stack, source, events)
            result = parallel.run_parallel_nested_prediction(*source, workers=1, resource_check=check)
            self.assertEqual(calls, Counter(pca=352, prepare=1, select=28, refit=28))
        self.assertEqual(tuple(result.predictions), nested.MODEL_IDS)
        self.assertEqual(result.outer_selections, serial.outer_selections)
        self.assertEqual(result.outer_cka_transforms, serial.outer_cka_transforms)
        self.assertEqual(result.final_predictors, serial.final_predictors)
        self.assertEqual(events.count(("derive_rank_pca", "after")), 1056)
        for model in nested.MODEL_IDS:
            self.assertEqual(tuple(result.outer_selections[model]), source[0].fold_labels)
            for group in nested.GROUPS:
                for stage in nested.STAGES:
                    np.testing.assert_array_equal(result.predictions[model][group][stage],
                                                  serial.predictions[model][group][stage])
                    self.assertFalse(result.predictions[model][group][stage].flags.writeable)

    def test_feature_or_model_failure_returns_no_complete_result(self):
        for phase in ("pca", "model"):
            source, events = inputs(), []
            with self.subTest(phase=phase), ExitStack() as stack:
                calls = self._instrument(stack, source, events, fail_pca=phase == "pca", fail_model=phase == "model")
                with self.assertRaises(ValueError):
                    parallel.run_parallel_nested_prediction(*source, workers=1,
                        resource_check=lambda event: events.append((event.operation, event.when)))
                self.assertNotIn(("complete", "after"), events)
                if phase == "pca":
                    self.assertEqual(calls["prepare"], 0)
                    self.assertEqual(calls["select"], 0)
                    self.assertNotIn(("all_features_feasible", "after"), events)
                else:
                    self.assertEqual(calls["prepare"], 1)
        with self.assertRaisesRegex(ValueError, "one_to_four"):
            parallel.run_parallel_nested_prediction(*source, workers=5, resource_check=lambda event: None)

    @unittest.skipUnless(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) >= 2,
                         "actual overlap check requires at least 2 allocated CPUs")
    def test_two_worker_overlap_canonical_collection_and_synchronized_guard(self):
        barrier, lock = Barrier(2, timeout=10), Lock()
        seen, active, callback_active = [], set(), []

        def callback(event):
            # This is deliberately not independently synchronized: the guard
            # owns callback serialization even when model tasks overlap.
            callback_active.append(event)
            self.assertEqual(len(callback_active), 1)
            # Release the GIL briefly so an unguarded concurrent callback
            # would overlap; four callbacks add only four milliseconds.
            time.sleep(.001)
            for _ in range(100):
                self.assertEqual(callback_active[0], event)
            callback_active.pop()

        guard = parallel.SynchronizedResourceGuard(callback)

        def task(model):
            with lock:
                active.add(model)
                seen.append(len(active))
            barrier.wait()
            guard(model)
            barrier.wait()
            with lock:
                active.remove(model)
            return model

        result = parallel._parallel_models(task, workers=2, guard=guard)
        self.assertEqual(tuple(result), nested.MODEL_IDS)
        self.assertEqual(max(seen), 2)
        self.assertEqual(active, set())
        guard.abort()
        with self.assertRaisesRegex(ValueError, "whole_run_aborted"):
            guard("no callback after failure")


if __name__ == "__main__":
    unittest.main()
