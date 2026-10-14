"""Inactive exact reuse of EP18 core fits with identical inner training blocks.

Alternative fit_person with the original call/retention/progress signatures.
The ordered splits (i,j) and (j,i) train on the SAME eight blocks in the SAME
order. Their original nuisance fit, five-family training PCA, whitening and
thirteen image fits can therefore be shared. Their validation designs, EEG,
sequence losses and nuisance denominators remain separate. No validation
response is used to choose the shared fit or a training transformation.

All fifteen unordered inner pairs are completed, then the original six outer
fits. Each outer choice uses its original increasing-inner-index summation and
strict loss/larger-ridge/index tie order. This retains thirty logical inner
plus six outer evaluations and 438 original logical multivariate fits, while
executing 228 distinct multivariate fits (each with the original 61 contrasts).
Only small inner loss records survive a pair; the two cache views share one
training state. No model, penalty, solver tolerance or numerical algebra changes.

No reader, CLI, resume, resource grant or scientific-release authority is
provided. Existing production sources and running callers are untouched.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import production_baseline_controller as controller
import production_core_reference as core
from training_transforms import CommonSensorSpace


def _counterpart(cache, sequences, sensor, convention, check):
    """Reverse only the held-out side; share every training-derived object."""
    split = cache.split
    core.require(split.inner_index is not None and split == folds._split(split.outer_index, split.inner_index),
                 "original_inner_split_required")
    reverse = folds._split(split.inner_index, split.outer_index)
    core.require(reverse.training_blocks == split.training_blocks,
                 "symmetric_training_blocks_must_be_identical")
    training = tuple(item for item in sequences if item.block in reverse.training_blocks)
    core.require(len(training) == len(cache.training) and all(a is b for a, b in zip(training, cache.training))
                 and cache.preparation.training_sequence_ids == tuple(item.sequence_id for item in training),
                 "symmetric_training_rows_and_order_must_be_identical")
    heldout = tuple(item for item in sequences if item.block in reverse.heldout_blocks)
    test = folds._designs(heldout, cache.preparation, core.CORE_MODELS, check)
    core.require(all(cache.train_designs[name].sequences[0].schema == test[name].sequences[0].schema
                     for name in core.CORE_MODELS), "counterpart_predictor_schema_must_match_training")
    targets = tuple(sensor.apply(item.sensor_samples, channel_order=item.channel_order) for item in heldout)
    result = replace(cache, split=reverse, heldout=heldout, test_designs=test, heldout_targets=targets,
                     nuisance_losses=np.empty(0))
    result.nuisance_losses = tuning._score(result, "nuisance", cache.nuisance_fit, check)
    tuning.normalize_losses(result.nuisance_losses, result.nuisance_losses,
                            normalization=convention.normalization)
    return result


def _inner_pair(first, second, convention, limits, check):
    """Fit every original penalty once; score the two validation sets separately."""
    core.require(first.split.outer_index == second.split.inner_index
                 and first.split.inner_index == second.split.outer_index
                 and first.split.training_blocks == second.split.training_blocks,
                 "exact_symmetric_inner_pair_required")
    for name in ("training", "preparation", "train_designs", "targets", "weights", "penalties",
                 "nuisance_fit", "whitener"):
        core.require(getattr(first, name) is getattr(second, name), "one_identical_training_state_required")
    errors = ([], [])
    for index in range(13):
        check()
        fit = folds._fit(first.train_designs["image"], first.targets, first.weights,
                         tuning._grid_penalty(first, "image", index), limits, check)
        for cache, values in zip((first, second), errors):
            values.append(tuning._score(cache, "image", fit, check))
        del fit
    result = []
    for cache, values in zip((first, second), errors):
        losses = folds._readonly(values)
        normalized = folds._readonly(tuning.normalize_losses(losses, cache.nuisance_losses,
                                                             normalization=convention.normalization))
        result.append(controller.CoreInnerResult(cache.split,
            tuple(item.sequence_id for item in cache.training), tuple(item.sequence_id for item in cache.heldout),
            losses, cache.nuisance_losses, normalized))
    return tuple(result)


def fit_person(person, recipe, *, check, retain_inner, retain_outer, progress):
    """Complete the unchanged logical panel using one inner training state at a time."""
    recipe = core.validate_recipe(recipe)
    sensor = CommonSensorSpace(person.inputs.channel_order)
    limits = folds.SolverLimits(**recipe["solver"])
    sequences = folds._roster(person.sequences, sensor, check)
    convention = SimpleNamespace(normalization=recipe["normalization"])
    progress.update(inner_splits_completed=0, outer_splits_completed=0,
                    unique_inner_training_sets_completed=0,
                    execution_algorithm="symmetric_inner_training_reuse")
    inner_by_split = {}
    for first_index in range(6):
        for second_index in range(first_index+1, 6):
            progress.update(outer_index=first_index, inner_index=second_index)
            cache = core._prepare_core(sequences, folds._split(first_index, second_index), sensor, recipe,
                person.prepare_training, limits, check, lambda value: progress.update(stage=value))
            progress["stage"] = "counterpart_validation_designs"
            counterpart = _counterpart(cache, sequences, sensor, convention, check)
            progress["stage"] = "paired_complete13_image_inner_grid"
            results = _inner_pair(cache, counterpart, convention, limits, check)
            # Results contain only paired losses/IDs; no design, targets, fitted
            # PCA, whitening, coefficients or training cache are retained here.
            del counterpart, cache
            for row in results:
                progress.update(outer_index=row.split.outer_index, inner_index=row.split.inner_index)
                retain_inner(row)
                inner_by_split[(row.split.outer_index, row.split.inner_index)] = row
                progress["inner_splits_completed"] += 1
            progress["unique_inner_training_sets_completed"] += 1
            del row, results
    core.require(set(inner_by_split) == {(i,j) for i in range(6) for j in range(6) if i != j},
                 "all_thirty_original_inner_evaluations_required")
    outer = []
    for outer_index in range(6):
        progress.update(outer_index=outer_index, inner_index=None)
        # Preserve the original reduction order, not unordered-pair completion order.
        inner = [inner_by_split[(outer_index, j)] for j in range(6) if j != outer_index]
        mean = np.mean([item.normalized_image_losses for item in inner], axis=0)
        choice = min(range(13), key=lambda index: (mean[index], -tuning.RIDGE_GRID[index], index))
        cache = core._prepare_core(sequences, folds._split(outer_index), sensor, recipe,
            person.prepare_training, limits, check, lambda value: progress.update(stage=value))
        progress["stage"] = "selected_outer_image_and_reduced_fits"
        fitted = controller._core_outer(cache, choice, convention, sensor, limits, check)
        retain_outer(fitted)
        outer.append({"outer_index": outer_index, "image_index": choice,
                      "normalized_losses": dict(fitted.result.normalized_losses)})
        progress["outer_splits_completed"] += 1
        del cache, fitted, inner
    # Counts describe the COMPLETE run only, not partially returned native calls.
    progress.update(actual_multivariate_fits=228, original_logical_multivariate_fits=438)
    return {"participant_id": person.participant_id, "outer": outer,
        "normalized_participant_losses": {name: float(np.mean([row["normalized_losses"][name] for row in outer]))
                                          for name in core.CORE_MODELS}}
