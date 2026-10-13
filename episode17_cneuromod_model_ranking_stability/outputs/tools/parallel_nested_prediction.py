"""Complete EP17 nested fitting, with bounded independent model threads.

The numerical kernels, grid, selector and refit are the existing implementation.
All 352 full PCA fits and their three exact rank cutoffs finish before response
qualification or selection. Responses/baselines are prepared once; each model
then completes all six outer fits and its final all-development fit. There is
no model subset, partial result, scoring release, file reader or retry here.

The caller admits one allocation, shared-input memory, concurrent workspaces,
deadline and publication. Numerical libraries must use one thread per worker.
Callbacks are serialized, but native kernels run concurrently. On any failure,
pending tasks are cancelled and running tasks refuse their next boundary; the
caller must also use Slurm and an external timeout for native kernels.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import RLock
from types import MappingProxyType

import numpy as np

import nested_prediction as nested


class ParallelPredictionError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ParallelPredictionError(code)


class SynchronizedResourceGuard:
    """Serialize resource accounting and propagate whole-run failure only."""
    def __init__(self, callback):
        require(callable(callback), "external_resource_callback_required")
        self.callback, self.lock, self.failed = callback, RLock(), False

    def abort(self):
        with self.lock:
            self.failed = True

    def __call__(self, event=None):
        with self.lock:
            require(not self.failed, "whole_run_aborted")
            try:
                self.callback(event)
            except BaseException:
                self.failed = True
                raise


def _parallel_models(function, *, workers, guard):
    def work(model):
        try:
            return function(model)
        except BaseException:
            guard.abort()
            raise

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ep17-model")
    futures = {}
    try:
        for model in nested.MODEL_IDS:
            futures[executor.submit(work, model)] = model
        completed = {}
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
        require(set(completed) == set(nested.MODEL_IDS), "all_four_model_tasks_required")
        return {model: completed[model] for model in nested.MODEL_IDS}
    except BaseException:
        guard.abort()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _model_pcas(model, features, training, check):
    bank = {}
    for folds, rows in training.items():
        for layer in nested.LAYERS:
            context = dict(model_id=model, training_folds=folds, layer=layer,
                           rank=max(nested.PCA_RANKS), shape=(len(rows), nested.FEATURE_WIDTHS[layer]))
            values = nested._slice_rows(check, "slice_features", features[model][layer], rows, context)
            maximum = nested._call(check, "fit_pca", nested.core.fit_pca, values,
                                   max(nested.PCA_RANKS), context=context)
            maximum = nested._call(check, "freeze_pca", nested._frozen_pca, maximum, context=context)
            bank[model, folds, layer] = {
                rank: nested._call(check, "derive_rank_pca", nested.derive_rank_pca,
                    maximum, values, rank, context={**context, "rank": rank})
                for rank in nested.PCA_RANKS}
    return bank


def _model_fit(model, design, features, responses, training, validation, pcas, means, check):
    bank = {group: {} for group in nested.GROUPS}
    for group in nested.GROUPS:
        for stage in nested.STAGES:
            nested._event(check, "allocate_OOF_bank", model_id=model, group=group, stage=stage,
                          shape=responses[group][stage].shape)
            bank[group][stage] = np.full_like(responses[group][stage], np.nan)
    outer, outer_cka = {}, {}
    for heldout, test_rows in validation.items():
        folds = tuple(label for label in design.fold_labels if label != heldout)
        selection = nested._select(model, folds, features, responses, training, validation, pcas, means, check)
        _, cka, _ = nested._refit(model, folds, selection, features, responses, training, pcas, means,
                                check, test_rows=test_rows, destination=bank)
        outer[heldout], outer_cka[heldout] = selection, cka
    for group in nested.GROUPS:
        for stage in nested.STAGES:
            nested.require(np.isfinite(bank[group][stage]).all(), "outer_prediction_missing_or_nonfinite")
            bank[group][stage] = nested._call(check, "freeze_OOF_bank", nested.endpoint.frozen_array,
                bank[group][stage], context=dict(model_id=model, group=group, stage=stage,
                                                 shape=bank[group][stage].shape))
    folds = design.fold_labels
    selection = nested._select(model, folds, features, responses, training, validation, pcas, means, check)
    pca, cka, ridges = nested._refit(model, folds, selection, features, responses, training, pcas, means, check)
    final = nested.FrozenPredictor(selection.setting, pca, cka, ridges, selection,
                                   design.image_ids, design.concept_ids)
    return bank, outer, outer_cka, final


def run_parallel_nested_prediction(design, features, responses, frozen_columns, *, workers, resource_check):
    """Return the complete original result type; all resource choices are explicit.

    workers may be 1..4, with no change to the four-model scientific scope.
    A single worker is useful for equivalence checks, not a smaller panel.
    Slurm allocation verification belongs to the calling compute entrypoint.
    """
    require(type(workers) is int and 1 <= workers <= len(nested.MODEL_IDS), "one_to_four_workers_required")
    require(all(os.environ.get(name) == "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
            and all(os.environ.get(name, "1") == "1" for name in ("VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")),
            "one_numerical_thread_per_worker_required")
    guard = SynchronizedResourceGuard(resource_check)
    nested._event(guard, "begin")
    try:
        features, responses, columns = nested._bind_inputs(design, features, responses, frozen_columns, guard)
        features, responses = nested._nested_map(features), nested._nested_map(responses)
        training, validation, blocks = nested._row_indices(design)
        for rows in (*training.values(), *validation.values(), *blocks.values()):
            rows.flags.writeable = False
        by_model = _parallel_models(lambda model: _model_pcas(model, features, training, guard),
                                    workers=workers, guard=guard)
        pcas = {key: value for model in nested.MODEL_IDS for key, value in by_model[model].items()}
        expected = {(model, folds, layer) for model in nested.MODEL_IDS
                    for folds in training for layer in nested.LAYERS}
        require(len(training) == 22 and len(expected) == 352 and set(pcas) == expected
                and all(set(value) == set(nested.PCA_RANKS) for value in pcas.values()),
                "complete_352_PCA_barrier_required")
        pcas = nested._nested_map(pcas)
        nested._event(guard, "all_features_feasible", "after")
        means, baselines = nested._prepare_responses(design, responses, training, validation, blocks, guard)
        means = nested._nested_map(means)
        completed = _parallel_models(lambda model: _model_fit(model, design, features, responses,
            training, validation, pcas, means, guard), workers=workers, guard=guard)
        result = nested.NestedPredictionResult(design, MappingProxyType(columns),
            nested._nested_map({model: completed[model][0] for model in nested.MODEL_IDS}),
            nested._nested_map(baselines),
            nested._nested_map({model: completed[model][1] for model in nested.MODEL_IDS}),
            nested._nested_map({model: completed[model][2] for model in nested.MODEL_IDS}),
            MappingProxyType({model: completed[model][3] for model in nested.MODEL_IDS}))
        nested._event(guard, "complete", "after")
        return result
    except BaseException:
        guard.abort()
        raise
