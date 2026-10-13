"""Five outcome-blind cost-pilot composition checks; coordinated Slurm only.

Keep the complete2423-row design and real feature widths. Instrument existing
PCA/ridge kernels; do not repeat their numerical qualification or read payloads.
Broadcast source fixtures poison withheld features and all nontraining neural
rows. Guards refuse forbidden slicing/casting before any materialization.
"""
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass, replace
from itertools import product
import io
import unittest
from unittest.mock import patch

import numpy as np

import development_prediction_cost as pilot
import nested_prediction as nested
import prediction_core as numerical


class GuardedMatrix(np.ndarray):
    def __new__(cls, value, *, kind, name, train, valid, positions, reads):
        result = value.view(cls)
        result.kind, result.name = kind, name
        result.train, result.valid, result.positions = train, valid, positions
        result.reads = reads
        return result

    def __array_finalize__(self, source):
        if source is not None:
            for name in ("kind", "name", "train", "valid", "positions", "reads"):
                setattr(self, name, getattr(source, name, None))

    def __getitem__(self, key):
        if self.kind == "feature":
            np.testing.assert_equal(isinstance(key, np.ndarray), True)
            admitted = np.array_equal(key, self.train) or np.array_equal(key, self.valid)
            if not admitted:
                raise AssertionError("forbidden feature rows, including fold01")
            stamp = (self.kind, self.name, "train" if np.array_equal(key, self.train) else "validation")
        else:
            if not isinstance(key, tuple) or len(key) != 2:
                raise AssertionError("targets must select rows and original-column positions together")
            np.testing.assert_array_equal(key[0], self.train[:, None])
            np.testing.assert_array_equal(key[1], self.positions[None, :])
            stamp = (self.kind, self.name, "training_and_fixed_columns")
        self.reads.append(stamp)
        # Only the validated slice becomes an ordinary numerical array.
        return np.ndarray.__getitem__(self, key).view(np.ndarray)


def fixture():
    concepts = tuple(item for item in range(480) for _ in range(6 if item < 23 else 5))
    design = nested.DevelopmentDesign(
        tuple(f"image_{index:04}" for index in range(2423)),
        tuple(f"concept_{item:03}" for item in concepts), ("development",) * 2423,
        tuple(f"fold{item // 80 + 1:02}" for item in concepts),
        tuple(f"block{item % 24 + 1:02}" for item in concepts))
    folds = np.asarray(design.fold_ids)
    train = np.flatnonzero(np.isin(folds, ("fold03", "fold04", "fold05", "fold06")))
    valid = np.flatnonzero(folds == "fold02")
    withheld = np.flatnonzero(folds == "fold01")
    reads, features, responses, columns, targets = [], {}, {}, {}, {}
    for mi, model in enumerate(nested.MODEL_IDS):
        layers = {}
        for li, layer in enumerate(nested.LAYERS):
            value = np.arange(2423, dtype=float) + (mi * 4 + li + 1) * 10000.
            value[withheld] = np.nan
            matrix = np.broadcast_to(value[:, None], (2423, nested.FEATURE_WIDTHS[layer]))
            layers[layer] = GuardedMatrix(matrix, kind="feature", name=(model, layer),
                train=train, valid=valid, positions=None, reads=reads)
        features[model] = nested.FeaturePanel(design, layers)
    permutation = np.array([19, 0, 18, 1, 17, 2, 16, 3, 15, 4, 14, 5, 13, 6, 12, 7, 11, 8, 10, 9])
    for gi, group in enumerate(nested.GROUPS):
        axis = gi * 1000 + 7 + 11 * permutation
        columns[group] = axis
        positions = np.argsort(axis)[:16]
        stages = {}
        for si, stage in enumerate(nested.STAGES):
            offset = gi * 100000. + si * 10000.
            value = np.arange(2423, dtype=float)[:, None] + offset + axis[None, :] / 1024.
            targets[group, stage] = value[np.ix_(train, positions)].copy()
            # Even finite-checking either closed neural fold is forbidden.
            value[np.r_[withheld, valid]] = np.nan
            value[:, np.argsort(axis)[16:]] = np.nan
            stages[stage] = GuardedMatrix(value, kind="response", name=(group, stage),
                train=train, valid=valid, positions=positions, reads=reads)
        responses[group] = nested.ResponsePanel(design, axis, stages)
    return design, features, responses, columns, train, valid, reads, targets


class DevelopmentPredictionCostTests(unittest.TestCase):
    def setUp(self):
        (self.design, self.features, self.responses, self.columns,
         self.train, self.valid, self.reads, self.targets) = fixture()
        self.current, self.seen_fits, self.seen_predictions = None, set(), set()
        self.last_fit = None

    def run_pilot(self, resource_check=None, **changes):
        arguments = dict(design=self.design, features=self.features,
                         responses=self.responses, frozen_columns=self.columns)
        arguments.update(changes)
        return pilot.measure_development_prediction_cost(**arguments,
            resource_check=self.resource if resource_check is None else resource_check)

    def resource(self, event):
        self.current = event

    def signature(self):
        event = self.current
        return (event.model_id, event.layer, event.rank, event.penalty, event.group, event.stage)

    def feature_base(self):
        return (nested.MODEL_IDS.index(self.current.model_id) * 4
                + nested.LAYERS.index(self.current.layer) + 1) * 10000.

    @contextmanager
    def kernels(self, *, fail_ridge=None, nonfinite_prediction=False):
        original_asarray = np.asarray
        original_array = np.array

        def guarded_cast(function, value, *args, **kwargs):
            if isinstance(value, GuardedMatrix):
                raise AssertionError("private source matrix was materialized before row/column admission")
            return function(value, *args, **kwargs)

        def fit_pca(value, rank):
            self.assertEqual((self.current.operation, self.current.when, rank), ("fit_pca", "before", 128))
            np.testing.assert_array_equal(value[:, 0], self.train + self.feature_base())
            self.assertEqual(value.shape, (1600, nested.FEATURE_WIDTHS[self.current.layer]))
            mean = np.zeros(value.shape[1])
            mean[0] = self.feature_base()
            return numerical.PCAFit(mean, np.zeros((value.shape[1], 128)), 1.,
                np.arange(value.shape[1], 0, -1, dtype=float), 128, len(value))

        def derive(maximum, value, rank):
            self.assertEqual(self.current.operation, "derive_rank_pca")
            self.assertEqual(maximum.feature_mean[0], self.feature_base())
            np.testing.assert_array_equal(value[:, 0], self.train + self.feature_base())
            return replace(maximum, loadings=maximum.loadings[:, :rank])

        def transform(fit, value):
            self.assertEqual(fit.feature_mean[0], self.feature_base())
            self.assertEqual(fit.rank, self.current.rank)
            rows = self.train if self.current.operation == "transform_training" else self.valid
            self.assertIn(self.current.operation, ("transform_training", "transform_validation"))
            np.testing.assert_array_equal(value[:, 0], rows + self.feature_base())
            return np.broadcast_to(rows.astype(float)[:, None], (len(rows), fit.rank))

        def fit_ridge(scores, targets, penalty):
            self.assertEqual(self.current.operation, "fit_ridge")
            np.testing.assert_array_equal(scores[:, 0], self.train)
            np.testing.assert_array_equal(targets, self.targets[self.current.group, self.current.stage])
            self.assertEqual(scores.shape, (1600, self.current.rank))
            self.assertEqual(targets.shape, (1600, 16))
            self.assertEqual(penalty, self.current.penalty)
            signature = self.signature()
            self.assertNotIn(signature, self.seen_fits)
            self.seen_fits.add(signature)
            if fail_ridge is not None and len(self.seen_fits) == fail_ridge:
                raise ValueError("PRIVATE_KERNEL_VALUE_7291_AND_PRIVATE_SOURCE_PATH")
            self.last_fit = numerical.RidgeFit(np.zeros(scores.shape[1]), np.zeros(16),
                np.zeros((scores.shape[1], 16)), penalty, len(scores))
            return self.last_fit

        def predict(fit, scores):
            self.assertEqual(self.current.operation, "predict_validation")
            self.assertIs(fit, self.last_fit)
            np.testing.assert_array_equal(scores[:, 0], self.valid)
            self.assertEqual(scores.shape, (400, self.current.rank))
            self.assertEqual(fit.penalty, self.current.penalty)
            self.seen_predictions.add(self.signature())
            return np.full((400, 16), np.nan if nonfinite_prediction else 0.)

        with ExitStack() as stack:
            # Plain replacements avoid Mock call histories retaining all8064
            # private target slices and defeating the bounded fixture memory.
            stack.enter_context(patch.object(np, "asarray", new=lambda value, *a, **k:
                guarded_cast(original_asarray, value, *a, **k)))
            stack.enter_context(patch.object(np, "array", new=lambda value, *a, **k:
                guarded_cast(original_array, value, *a, **k)))
            stack.enter_context(patch.object(numerical, "fit_pca", new=fit_pca))
            stack.enter_context(patch.object(nested, "derive_rank_pca", new=derive))
            stack.enter_context(patch.object(numerical.PCAFit, "transform", transform))
            stack.enter_context(patch.object(numerical, "fit_ridge", new=fit_ridge))
            stack.enter_context(patch.object(numerical.RidgeFit, "predict", predict))
            for module, name in ((numerical, "error_sums"), (numerical, "pooled_r2"),
                                 (numerical, "hyperparameter_tie_key"), (numerical, "linear_cka"),
                                 (nested, "pooled_inner_objective"), (nested, "_select")):
                stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError("validation scores and selection are forbidden")))
            yield

    def assert_public_report(self, report):
        self.assertIsInstance(report, pilot.PredictionCostReport)
        expected = {"status", "failure_code", "last_operation", "completed_counts", "attempted_counts",
                    "operation_seconds", "total_wall_seconds", "process_cpu_seconds",
                    "process_peak_rss_kib", "dimensions"}
        self.assertEqual({field.name for field in fields(report)}, expected)

        def check(value):
            self.assertNotIsInstance(value, (np.ndarray, BaseException))
            if is_dataclass(value):
                for field in fields(value):
                    check(getattr(value, field.name))
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    self.assertIsInstance(key, str)
                    check(item)
            else:
                self.assertIsInstance(value, (str, int, float, type(None)))
        check(report)
        self.assertNotIn("PRIVATE_KERNEL_VALUE", repr(report))
        self.assertNotIn("PRIVATE_BUDGET_DETAILS", repr(report))
        self.assertGreaterEqual(report.total_wall_seconds, 0.)
        self.assertGreaterEqual(report.process_cpu_seconds, 0.)
        self.assertGreater(report.process_peak_rss_kib, 0)

    def test_complete_grid_joins_original_columns_and_never_reads_closed_rows_or_scores(self):
        with self.kernels():
            result = self.run_pilot()
        self.assertEqual((result.status, result.failure_code), ("complete", None))
        expected = dict(bind_metadata=1, slice_features=32, fit_pca=16, derive_rank_pca=48,
                        transform_training=48, transform_validation=48,
                        slice_targets=8064, fit_ridge=8064, predict_validation=8064)
        self.assertEqual(dict(result.completed_counts), expected)
        self.assertEqual(dict(result.attempted_counts), expected)
        opportunities = set(product(nested.MODEL_IDS, nested.LAYERS, nested.PCA_RANKS,
                                    nested.RIDGE_PENALTIES, nested.GROUPS, nested.STAGES))
        self.assertEqual(self.seen_fits, opportunities)
        self.assertEqual(self.seen_predictions, opportunities)
        self.assertEqual(len(self.reads), 32 + 8064)
        self.assertEqual((result.dimensions["training_images"], result.dimensions["validation_images"],
                          result.dimensions["withheld_fold01_images"]), (1600, 400, 423))
        self.assertEqual(result.dimensions["target_columns_per_group"], 16)
        self.assert_public_report(result)

    def test_invalid_metadata_refuses_before_any_feature_or_response_value_access(self):
        group = nested.GROUPS[-1]
        cases = [dict(features={key: panel for key, panel in self.features.items() if key != nested.MODEL_IDS[-1]}),
                 dict(features={**self.features, nested.MODEL_IDS[-1]:
                                replace(self.features[nested.MODEL_IDS[-1]], rows=None)}),
                 dict(responses={**self.responses, group:
                                replace(self.responses[group], column_ids=self.columns[group][::-1])}),
                 dict(responses={**self.responses, group:
                                replace(self.responses[group], stages={"B": self.responses[group].stages["B"]})}),
                 dict(frozen_columns={**self.columns, group: self.columns[group][:15]})]
        with self.kernels():
            for changed in cases:
                result = self.run_pilot(**changed)
                self.assertEqual((result.status, result.failure_code), ("stopped", "metadata_refused"))
                self.assertEqual(self.reads, [])
                self.assertEqual(result.completed_counts["bind_metadata"], 0)
                self.assertEqual(sum(result.completed_counts.values()), 0)
                self.assert_public_report(result)

    def test_resource_interruption_keeps_honest_counts_without_partial_readiness(self):
        for when in ("before", "after"):
            self.seen_fits.clear()
            self.seen_predictions.clear()
            count = 0

            def budget(event):
                nonlocal count
                self.resource(event)
                if event.operation == "fit_ridge" and event.when == when:
                    count += 1
                    if count == (4 if when == "before" else 3):
                        raise RuntimeError("PRIVATE_BUDGET_DETAILS")

            with self.kernels():
                result = self.run_pilot(resource_check=budget)
            self.assertEqual((result.status, result.failure_code, result.last_operation),
                             ("stopped", "resource_refused", "fit_ridge"))
            self.assertEqual(result.completed_counts["fit_ridge"], 3)
            self.assertEqual(result.attempted_counts["fit_ridge"], 3)
            self.assertEqual(result.completed_counts["predict_validation"], 3 if when == "before" else 2)
            self.assertLess(result.completed_counts["fit_ridge"], 8064)
            self.assert_public_report(result)

    def test_private_kernel_errors_and_nonfinite_predictions_release_no_arrays_or_error_text(self):
        for nonfinite in (False, True):
            self.seen_fits.clear()
            self.seen_predictions.clear()
            output = io.StringIO()
            with self.kernels(fail_ridge=None if nonfinite else 4, nonfinite_prediction=nonfinite), \
                    redirect_stdout(output), redirect_stderr(output):
                result = self.run_pilot()
            self.assertEqual(output.getvalue(), "")
            self.assertEqual((result.status, result.failure_code), ("stopped", "operation_failed"))
            operation = "predict_validation" if nonfinite else "fit_ridge"
            self.assertEqual(result.last_operation, operation)
            self.assertEqual(result.attempted_counts[operation], 1 if nonfinite else 4)
            self.assertEqual(result.completed_counts[operation], 0 if nonfinite else 3)
            self.assert_public_report(result)

    def test_resource_callback_is_required_before_metadata_or_value_work(self):
        result = pilot.measure_development_prediction_cost(self.design, self.features,
            self.responses, self.columns, resource_check=None)
        self.assertEqual((result.status, result.failure_code), ("stopped", "resource_callback_required"))
        self.assertIsNone(result.last_operation)
        self.assertEqual(sum(result.attempted_counts.values()), 0)
        self.assertEqual(sum(result.completed_counts.values()), 0)
        self.assertEqual(self.reads, [])
        self.assert_public_report(result)


if __name__ == "__main__":
    unittest.main()
