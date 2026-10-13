"""Complete conditional observed-beta losses from EP15's existing fit records.

No fitting, reader, source replacement, uncertainty, ratio or class decision.
Each fixed control draw is squared before its losses are averaged. The returned
12 x 4 x 18 contributions preserve every participant, partition and condition;
they are not independent biological replicates. Access and publication remain
the coordinator's responsibility under the adopted observed-beta route.
"""
import numpy as np

from calibration_falsifiers import (
    CalibrationFalsifierRun, VARIANTS, _panel_predictions, identity_pairings,
)
from development_lopo import (
    DEVELOPMENT_IDS, MATCHED_WARP_DRAWS, RANDOM_BASIS_DRAWS, M2_VARIANTS,
    M3_VARIANTS, CalibrationReference, FoldContext, LOPORun, PairedIdentityContext,
    _array, _support_size, validate_source_context,
)
from local_functional_source import TASK_A_CONDITIONS
from registered_endpoint_metrics import CONDITIONS, condition_weights


ANALYSIS_SCOPE = "conditional_observed_beta"
ROUTE_ID = "observed_beta_development_descriptive"
PARTITION_RUN_IDS = tuple(tuple(range(start, start + 4)) for start in (1, 5, 9, 13))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_complete_join(ordinary, controls, *, support_size=208):
    """Check the existing whole-panel identities before converting any map."""
    _require(type(ordinary) is LOPORun and type(controls) is CalibrationFalsifierRun,
             "complete ordinary and calibration-control fit records are required")
    for run in (ordinary, controls):
        _require(run.deleted_id is None and run.participant_ids == DEVELOPMENT_IDS,
                 "the complete fixed twelve-person run is required")
    for folds in (ordinary.folds, controls.condition_folds, controls.identity_folds):
        _require(len(folds) == 12 and tuple(f.context.target_id for f in folds) == DEVELOPMENT_IDS,
                 "all twelve targets must remain in canonical order")
    donors = dict(identity_pairings())
    warp_keys = tuple((name, draw) for name in M2_VARIANTS for draw in range(MATCHED_WARP_DRAWS))
    voxel_ids = None
    for fold, condition, identity in zip(ordinary.folds, controls.condition_folds, controls.identity_folds):
        _require(type(fold.context) is FoldContext and fold.context.deleted_id is None,
                 "ordinary fitting must retain its original LOPO source")
        validate_source_context(fold.context)
        reference = fold.calibration_reference
        _require(type(reference) is CalibrationReference and reference.context == fold.context,
                 "ordinary genuine-A calibration reference must remain attached")
        source = reference.source
        if voxel_ids is None:
            voxel_ids = source.voxel_ids
        _require(len(voxel_ids) == support_size and len(set(voxel_ids)) == support_size
                 and source.voxel_ids == voxel_ids and source.task_a_condition_ids == TASK_A_CONDITIONS
                 and source.training_ids == fold.context.training_ids,
                 "ordinary source must retain exact training and condition/location axes")
        _prediction_values(fold)
        _panel_predictions(reference.panel)
        _require(fold.m0a_prediction is not None
                 and fold.m0a_prediction.training_ids == fold.context.training_ids,
                 "M0a must retain each ordinary source's training exclusions")
        _require(tuple(value.draw_index for value in fold.random_basis_predictions)
                 == tuple(range(RANDOM_BASIS_DRAWS)), "all eight fixed random-basis draws are required")
        _require(tuple((value.variant, value.draw_index) for value in fold.matched_warp_predictions)
                 == warp_keys, "all four variants and eight matched-warp draws are required")
        for control, paired in ((condition, False), (identity, True)):
            context = control.context
            _require(type(context) is (PairedIdentityContext if paired else FoldContext)
                     and context.deleted_id is None and context.target_id == fold.context.target_id,
                     "the condition/identity target must preserve its own source design")
            validate_source_context(context)
            _require(control.source.training_ids == context.training_ids
                     and control.source.voxel_ids == voxel_ids
                     and control.source.task_a_condition_ids == TASK_A_CONDITIONS
                     and control.true_packet_id == context.target_id
                     and control.control_packet_id == (context.donor_id if paired else context.target_id),
                     "control packets and source identities must preserve the paired design")
            _panel_predictions(control.true_panel)
            _panel_predictions(control.control_panel)
            if paired:
                _require(context.donor_id == donors[context.target_id],
                         "the original fixed recipient/donor cycle is required")
                _require(control.true_panel is not reference.panel,
                         "identity true calibration must use its paired reduced source")
            else:
                _require(context == fold.context and control.source is source
                         and control.true_panel is reference.panel,
                         "condition control must reuse the ordinary true panel and source")
    return voxel_ids


def _prediction_values(fold):
    """FoldPrediction and CalibrationPanel share these four canonical fields."""
    from calibration_falsifiers import VARIANT_FIELDS
    values = []
    for field, names in VARIANT_FIELDS:
        predictions = getattr(fold, field)
        _require(tuple(value.name for value in predictions) == names,
                 "all sixteen variants must remain in canonical order")
        values.extend(predictions)
    return tuple(values)


def _reduce(values, *, difference=False):
    values = _array(values, (12, 4, 18), "observed-loss contributions")
    per_person = np.mean(np.sum(values * condition_weights()[None, None, :], axis=-1), axis=1)
    prefix = "mse_difference" if difference else "observed_mse"
    return {"partition_condition_" + prefix: values,
            "participant_mean_" + prefix: per_person,
            "mean_" + prefix: float(np.mean(per_person))}


def _loss(observed_centered, maps):
    support_size = observed_centered.shape[-1]
    maps = np.stack([_array(value, (18, support_size), "prediction maps") for value in maps])
    centered = maps - maps.mean(axis=-1, keepdims=True)
    squared = (observed_centered - centered[:, None]) ** 2
    return _reduce(np.mean(squared, axis=-1))


def _difference(left, right):
    return _reduce(left["partition_condition_observed_mse"]
                   - right["partition_condition_observed_mse"], difference=True)


def _draw_summary(draws, true_loss):
    # The input is already one loss tensor per draw. Never average predictions.
    mean = _reduce(np.mean([draw["partition_condition_observed_mse"] for draw in draws], axis=0))
    return {"draw_losses": tuple(draws), "mean_draw_loss": mean,
            "mean_control_minus_true": _difference(mean, true_loss)}


def observed_beta_summary(ordinary, controls, *, check, support_size=208):
    """Score every existing fixed prediction and its source-matched controls."""
    _require(callable(check), "an explicit resource callback is required")
    check()
    support_size = _support_size(support_size)
    voxel_ids = _validate_complete_join(ordinary, controls, support_size=support_size)
    observed = np.stack([_array(fold.observed_partitions, (4, 18, support_size), "observed Task B")
                         for fold in ordinary.folds])
    centered = observed - observed.mean(axis=-1, keepdims=True)
    ordinary_predictions = tuple(_prediction_values(fold) for fold in ordinary.folds)
    # The ordinary fold holds copies of the already fitted reference maps.
    for fold, predictions in zip(ordinary.folds, ordinary_predictions):
        check()
        _require(np.array_equal(fold.G, fold.calibration_reference.source.G),
                 "ordinary M0b must remain its own fitted source mean")
        for prediction, fitted in zip(predictions, _panel_predictions(fold.calibration_reference.panel)):
            _require(np.array_equal(prediction.maps, fitted.maps),
                     "ordinary prediction and reused true calibration panel changed")
    losses = {"M0a": _loss(centered, [fold.m0a_prediction.maps for fold in ordinary.folds]),
              "M0b": _loss(centered, [fold.G for fold in ordinary.folds])}
    for index, name in enumerate(VARIANTS):
        check()
        losses[name] = _loss(centered, [values[index].maps for values in ordinary_predictions])
    comparisons = {name: {"model_minus_" + baseline: _difference(losses[name], losses[baseline])
                          for baseline in ("M0a", "M0b")} for name in VARIANTS}
    basis_draws = []
    for draw in range(RANDOM_BASIS_DRAWS):
        check()
        basis_draws.append(_loss(centered, [fold.random_basis_predictions[draw].maps
                                           for fold in ordinary.folds]))
    basis = _draw_summary(basis_draws, losses[M3_VARIANTS[0]])
    basis["matched_true_variant"] = M3_VARIANTS[0]
    warps = {}
    for index, variant in enumerate(M2_VARIANTS):
        draws = []
        for draw in range(MATCHED_WARP_DRAWS):
            check()
            draws.append(_loss(centered, [fold.matched_warp_predictions[index * MATCHED_WARP_DRAWS + draw].maps
                                         for fold in ordinary.folds]))
        warps[variant] = _draw_summary(draws, losses[variant])
    calibration = {}
    for label, folds in (("condition", controls.condition_folds), ("identity", controls.identity_folds)):
        true = tuple(_panel_predictions(fold.true_panel) for fold in folds)
        control = tuple(_panel_predictions(fold.control_panel) for fold in folds)
        variants = {}
        for index, name in enumerate(VARIANTS):
            check()
            true_loss = _loss(centered, [values[index].maps for values in true])
            control_loss = _loss(centered, [values[index].maps for values in control])
            variants[name] = {"true": true_loss, "control": control_loss,
                              "control_minus_true": _difference(control_loss, true_loss)}
        calibration[label] = {"source_training_ids": tuple(fold.context.training_ids for fold in folds),
            "true_packet_ids": tuple(fold.true_packet_id for fold in folds),
            "control_packet_ids": tuple(fold.control_packet_id for fold in folds),
            "variants": variants,
            "comparison": ("donor_minus_true_within_paired_reduced_source" if label == "identity"
                           else "relabeled_minus_true_within_ordinary_source")}
    check()
    return {"analysis_scope": ANALYSIS_SCOPE, "route_id": ROUTE_ID, "biological_n": 12,
        "participant_ids": DEVELOPMENT_IDS, "condition_ids": CONDITIONS,
        "partition_run_ids": PARTITION_RUN_IDS, "voxel_ids": voxel_ids,
        "condition_weights": condition_weights(), "participant_weight": 1 / 12,
        "partition_weight": 1 / 4, "voxel_weight": 1 / support_size, "support_size": support_size,
        "contribution_axes": ("participant", "partition", "condition"),
        "contribution_definition": "mean_voxel_squared_error_after_separate_spatial_centering",
        "aggregation": "equal_partitions_then_seven_domain_weighting_then_equal_participants",
        "interpretation": "observed_beta_prediction_error_including_measurement_error",
        "uncertainty_calibrated": False, "formal_class_decision": False,
        "ordinary": {"source_training_ids": tuple(f.context.training_ids for f in ordinary.folds),
                     "losses": losses, "model_minus_baseline": comparisons},
        "random_local_basis": basis, "matched_random_warp": warps,
        "calibration_controls": calibration}
