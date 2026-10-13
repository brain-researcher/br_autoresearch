"""The specified EP17 outcome-blind prediction-cost pilot, when selected.

Use complete caller-admitted development panels, but withhold fold01 entirely:
train on fold03..fold06 and predict fold02, without reading any validation
response. Fit all four models, four layers, three ranks, seven penalties and
all eight participant/ROI groups at B/C/D. The only target restriction is the
specified sixteen smallest original A_all column IDs per group. No response or
feature value selects this subset. This is not a smaller scientific experiment.

Reuse exactly the production PCA/ridge/prediction kernels, including one full
max128 SVD per model/layer and rank-specific scaling. Discard coefficients and
predictions; never calculate errors, objectives, validation scores or a winner.
Return dimensions, operation counts, timing and process-lifetime peak RSS only.

This caller performs no source reads, cache persistence or reader release. The
pre-score freeze, admitted panels, actual isolation and measured resource/job
selection remain external. The callback must stop work beyond the coordinator's
limits, with an external timeout for a blocked numerical call. A hard kill may
prevent any report; caught failures return a stopped report with fixed codes.
One partition with sixteen target columns per group does not certify memory or
runtime for the full 22-partition, all-column fit. This module neither extrapolates
resources automatically nor activates that full fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import resource
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as core


FOLDS = ("fold01", "fold02", "fold03", "fold04", "fold05", "fold06")
TRAINING_FOLDS = FOLDS[2:]
TARGET_COLUMNS = 16
OPERATIONS = ("bind_metadata", "slice_features", "fit_pca", "derive_rank_pca",
              "transform_training", "transform_validation", "slice_targets",
              "fit_ridge", "predict_validation")
REQUIRED_COUNTS = MappingProxyType({"fit_pca": 16, "derive_rank_pca": 48,
    "transform_training": 48, "transform_validation": 48,
    "fit_ridge": 8064, "predict_validation": 8064})


class _MetadataRefused(ValueError):
    pass


class _ResourceRefused(RuntimeError):
    pass


def _require(condition):
    if not condition:
        raise _MetadataRefused("fixed pilot metadata do not match")


@dataclass(frozen=True)
class ResourceEvent:
    operation: str
    when: str = "before"
    model_id: str | None = None
    layer: str | None = None
    rank: int | None = None
    penalty: float | None = None
    group: tuple | None = None
    stage: str | None = None
    shape: tuple = ()


@dataclass(frozen=True)
class PredictionCostReport:
    status: str
    failure_code: str | None
    last_operation: str | None
    completed_counts: Mapping
    attempted_counts: Mapping
    operation_seconds: Mapping
    total_wall_seconds: float
    process_cpu_seconds: float
    process_peak_rss_kib: int
    dimensions: Mapping


class _Measurement:
    def __init__(self, check):
        self.check = check
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.last = None
        self.completed = dict.fromkeys(OPERATIONS, 0)
        self.attempted = dict.fromkeys(OPERATIONS, 0)
        self.seconds = dict.fromkeys(OPERATIONS, 0.)
        self.dimensions = {}

    def event(self, operation, when, context):
        try:
            self.check(ResourceEvent(operation, when, **context))
        except Exception:
            raise _ResourceRefused("external resource callback refused") from None

    def call(self, operation, function, *args, context=None):
        context = {} if context is None else context
        self.last = operation
        self.event(operation, "before", context)
        self.attempted[operation] += 1
        started = time.perf_counter()
        try:
            value = function(*args)
        finally:
            # A failed operation consumed time but is not completed.
            self.seconds[operation] += time.perf_counter() - started
        self.completed[operation] += 1
        # Preserve that completed work if this AFTER callback stops the pilot.
        self.event(operation, "after", context)
        return value

    def report(self, status, code):
        return PredictionCostReport(status, code, self.last,
            MappingProxyType(dict(self.completed)), MappingProxyType(dict(self.attempted)),
            MappingProxyType(dict(self.seconds)), time.perf_counter() - self.started,
            time.process_time() - self.cpu_started,
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            MappingProxyType(dict(self.dimensions)))


def _keys(value, expected):
    _require(isinstance(value, Mapping) and set(value) == set(expected))


def _array_header(value, shape):
    # No cast, finite scan, full snapshot or value indexing at this stage.
    _require(isinstance(value, np.ndarray) and value.shape == shape
             and value.dtype.kind in "iuf")


def _bind(design, features, responses, frozen_columns):
    _require(isinstance(design, nested.DevelopmentDesign)
             and set(design.fold_ids) == set(FOLDS))
    _keys(features, nested.MODEL_IDS)
    _keys(responses, endpoint.GROUPS)
    _keys(frozen_columns, endpoint.GROUPS)
    for model in nested.MODEL_IDS:
        panel = features[model]
        _require(isinstance(panel, nested.FeaturePanel) and panel.rows == design)
        _keys(panel.layers, nested.LAYERS)
        for layer in nested.LAYERS:
            _array_header(panel.layers[layer], (nested.DEVELOPMENT_IMAGES, nested.FEATURE_WIDTHS[layer]))
    column_positions = {}
    for group in endpoint.GROUPS:
        panel = responses[group]
        _require(isinstance(panel, nested.ResponsePanel) and panel.rows == design)
        _keys(panel.stages, endpoint.STAGES)
        frozen = endpoint.column_ids(frozen_columns[group], "frozen_A_all")
        supplied = endpoint.column_ids(panel.column_ids, "response_columns")
        _require(len(frozen) >= TARGET_COLUMNS and np.array_equal(frozen, supplied)
                 and bool((frozen < endpoint.support.PARTICIPANT_COLUMNS[group[0]]).all()))
        # Original column IDs choose the fixed pilot support; response-array
        # positions can be arbitrarily ordered and must be joined explicitly.
        column_positions[group] = np.argsort(frozen)[:TARGET_COLUMNS]
        for stage in endpoint.STAGES:
            _array_header(panel.stages[stage], (nested.DEVELOPMENT_IMAGES, len(frozen)))
    train = np.asarray([i for i, fold in enumerate(design.fold_ids) if fold in TRAINING_FOLDS], dtype=np.int64)
    valid = np.asarray([i for i, fold in enumerate(design.fold_ids) if fold == "fold02"], dtype=np.int64)
    withheld = sum(fold == "fold01" for fold in design.fold_ids)
    _require(len({design.concept_ids[i] for i in train}) == 320
             and len({design.concept_ids[i] for i in valid}) == 80
             and len(train) + len(valid) + withheld == nested.DEVELOPMENT_IMAGES)
    return train, valid, withheld, column_positions


def _feature_rows(values, rows):
    # Select admitted rows BEFORE core.matrix converts to float64/checks finite.
    return core.matrix(values[rows], "cost_pilot_features", minimum_rows=2)


def _target_rows(values, rows, positions):
    # No validation-response access, even to check its values for finiteness.
    return core.matrix(values[np.ix_(rows, positions)], "cost_pilot_targets", minimum_rows=2)


def _predict(fit, scores, count):
    predicted = fit.predict(scores)
    if (not isinstance(predicted, np.ndarray) or predicted.shape != (count, TARGET_COLUMNS)
            or not np.isfinite(predicted).all()):
        raise ValueError("cost pilot prediction incomplete or nonfinite")
    # The caller discards this array immediately; it never reaches the report.
    return predicted


def _run(design, features, responses, frozen_columns, measurement):
    train, valid, withheld, positions = measurement.call(
        "bind_metadata", _bind, design, features, responses, frozen_columns)
    measurement.dimensions.update(training_images=len(train), validation_images=len(valid),
        withheld_fold01_images=withheld, training_concepts=320, validation_concepts=80,
        withheld_concepts=80, models=4, layers_per_model=4, settings_per_model=84,
        participant_ROI_groups=8, beta_stages=3, target_columns_per_group=TARGET_COLUMNS)
    for model in nested.MODEL_IDS:
        for layer in nested.LAYERS:
            context = dict(model_id=model, layer=layer)
            train_features = measurement.call("slice_features", _feature_rows,
                features[model].layers[layer], train,
                context={**context, "shape": (len(train), nested.FEATURE_WIDTHS[layer])})
            validation_features = measurement.call("slice_features", _feature_rows,
                features[model].layers[layer], valid,
                context={**context, "shape": (len(valid), nested.FEATURE_WIDTHS[layer])})
            maximum = measurement.call("fit_pca", core.fit_pca, train_features, max(nested.PCA_RANKS),
                context={**context, "rank": max(nested.PCA_RANKS), "shape": train_features.shape})
            for rank in nested.PCA_RANKS:
                ranked = {**context, "rank": rank}
                pca = measurement.call("derive_rank_pca", nested.derive_rank_pca,
                    maximum, train_features, rank, context=ranked)
                train_scores = measurement.call("transform_training", pca.transform, train_features,
                    context={**ranked, "shape": (len(train), rank)})
                validation_scores = measurement.call("transform_validation", pca.transform, validation_features,
                    context={**ranked, "shape": (len(valid), rank)})
                for penalty in nested.RIDGE_PENALTIES:
                    for group in endpoint.GROUPS:
                        for stage in endpoint.STAGES:
                            fitted = {**ranked, "penalty": penalty, "group": group, "stage": stage}
                            targets = measurement.call("slice_targets", _target_rows,
                                responses[group].stages[stage], train, positions[group],
                                context={**fitted, "shape": (len(train), TARGET_COLUMNS)})
                            fit = measurement.call("fit_ridge", core.fit_ridge, train_scores, targets, penalty,
                                context={**fitted, "shape": targets.shape})
                            predicted = measurement.call("predict_validation", _predict, fit,
                                validation_scores, len(valid),
                                context={**fitted, "shape": (len(valid), TARGET_COLUMNS)})
                            del predicted, fit, targets
                del pca, train_scores, validation_scores
            del maximum, train_features, validation_features


def measure_development_prediction_cost(design, features, responses, frozen_columns, *, resource_check):
    """Measure the specified full-grid pilot without selecting a model or job.

    The report's status must be inspected: stopped attempts retain honest partial
    counts and failed-operation elapsed time. Its fixed failure codes never
    include exception objects, source values or error text from private kernels.
    Counted predictions passed finite/shape checks; an incomplete count never
    certifies unvisited predictions. The process_peak_rss_kib field is the Linux
    process-lifetime high-water mark, including caller-resident full input panels
    and prior work; it is not incremental or isolated pilot memory. Total wall
    time includes callbacks/metadata, whereas operation_seconds times only calls.

    No callback can preempt a blocked SVD/solve. The caller must use the selected
    external runtime limit and one-thread allocation. No hard process termination
    is converted into a fictitious returned report. Full nested prediction still
    requires separately selected resources and all existing data/claim boundaries.
    """
    measurement = _Measurement(resource_check)
    if not callable(resource_check):
        return measurement.report("stopped", "resource_callback_required")
    try:
        _run(design, features, responses, frozen_columns, measurement)
        if any(measurement.completed[name] != count for name, count in REQUIRED_COUNTS.items()):
            return measurement.report("stopped", "incomplete_required_grid")
    except _MetadataRefused:
        return measurement.report("stopped", "metadata_refused")
    except _ResourceRefused:
        return measurement.report("stopped", "resource_refused")
    except Exception:
        code = "metadata_refused" if measurement.last == "bind_metadata" else "operation_failed"
        return measurement.report("stopped", code)
    return measurement.report("complete", None)
