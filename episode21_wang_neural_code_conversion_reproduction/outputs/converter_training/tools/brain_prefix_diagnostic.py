"""Caller-admitted first-fold diagnosis, with no fitting or empirical CLI.

This replays only the exact prefix of fit_brain_pair for the failed first pair.
The caller owns loading, categorical failure reporting and finite resources.
Passing this prefix does not localize the previous failure or qualify a model.
"""
from __future__ import annotations

import numpy as np

import brain_loss as brain
from empirical_brain_inputs import BoundBrainInputs, _admit


def diagnose_brain_first_fold(bound: BoundBrainInputs):
    if not isinstance(bound, BoundBrainInputs):
        raise ValueError("caller_admitted_bound_brain_inputs_required")
    source, target = bound.source.participant, bound.target.participant
    if ((source, target) != ("sub-03", "sub-01")
            or len(bound.source.native_voxel_ids) != 12897
            or len(bound.target.native_voxel_ids) != 12295):
        raise ValueError("original_failed_pair_and_complete_native_axes_required")
    result = bound.require_admitted_context(bound.execution_context,
        operation="diagnose_brain_first_fold", source=source, target=target,
        budget=bound.budget, execution_options=None)
    if result is not None:
        raise ValueError("external_resolver_must_raise_or_return_None")
    _admit(bound.execution_context, bound.require_admitted_context, source, target,
           bound.budget, bound.allocation_identity)
    check = bound.budget.check
    check()
    brain._validate_folds(bound.folds)
    x, _ = brain._aligned_values(bound.source, bound.folds, bound.source.native_voxel_ids)
    y, _ = brain._aligned_values(bound.target, bound.folds, bound.target.native_voxel_ids)
    check()
    validation = np.asarray(bound.folds.validation_indices[0])
    training_mask = np.ones(brain.TRAINING_IMAGES, dtype=bool)
    training_mask[validation] = False
    x_fit, y_fit = x[training_mask], y[training_mask]
    x_mean, x_std = brain._moments(x_fit, "source_fold_0")
    y_mean, y_std = brain._moments(y_fit, "target_fold_0")
    x_fit = brain._normalize(x_fit, x_mean, x_std, "source_fit")
    y_fit = brain._normalize(y_fit, y_mean, y_std, "target_fit")
    x_validation = brain._normalize(x[validation], x_mean, x_std, "source_validation")
    y_validation = brain._normalize(y[validation], y_mean, y_std, "target_validation")
    check()
    del x_fit, y_fit, x_validation, y_validation
    return {"status": "not_localized", "completed_prefix": "first_fold_normalization",
            "fold": 0, "fits_performed": 0, "scores_produced": False,
            "model_published": False}
