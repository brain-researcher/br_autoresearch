"""EP15 adopted M2 versus matched-warp alignment point statistic.

Consumes already fitted complete LOPO predictions; no fitting, reader or
empirical access. This is the finite-displacement source-map signature fixed
in SEARCH_POLICY, not a new gradient-linearization estimator. It supplies no
half-field stability certificate, inference, margin or M2 class decision.
"""
import numpy as np

from development_lopo import (
    DEVELOPMENT_IDS, MATCHED_WARP_DRAWS, M2_VARIANTS,
    FoldContext, M2Prediction, MatchedWarpPrediction, _array, _support_size, validate_source_context,
)
from registered_endpoint_metrics import condition_weights


def m2_alignment_points(run, *, support_size=208):
    """Equal-person alignment advantage for every fixed M2 variant.

    The observed pattern is each recipient B half minus its fold's G. The
    spatial field pattern is fitted prediction/common_gain minus G, recovering
    the source map under the field before the common Task-A gain. Matched
    fields use the identical gain removal, baseline and observed residual.
    Keeping gain in the pattern would reweight people by their fitted gains
    and would change the registered primary statistic.

    Average both observed halves, then the eight control-draw statistics,
    then participants, with seven-domain/equal-voxel inner products. Draws,
    conditions and halves are not independent biological units. Every required
    fold, variant and draw is retained; invalid or incomplete inputs refuse.
    """
    support_size = _support_size(support_size)
    if run.deleted_id not in (None, *DEVELOPMENT_IDS):
        raise ValueError("only an explicit development participant deletion is permitted")
    expected = tuple(person for person in DEVELOPMENT_IDS if person != run.deleted_id)
    keys = tuple((variant, draw) for variant in M2_VARIANTS for draw in range(MATCHED_WARP_DRAWS))
    if (run.participant_ids != expected or len(run.folds) != len(expected)
            or tuple(fold.context.target_id for fold in run.folds) != expected):
        raise ValueError("all retained ordered development folds are required")
    # All fold/variant/draw identities are checked before converting any maps.
    for fold in run.folds:
        if type(fold.context) is not FoldContext or fold.context.deleted_id != run.deleted_id:
            raise ValueError("alignment must retain the ordinary LOPO source context")
        validate_source_context(fold.context)
        if (any(type(value) is not M2Prediction for value in fold.m2_predictions)
                or tuple(value.name for value in fold.m2_predictions) != M2_VARIANTS):
            raise ValueError("all four M2 variants are required in fixed order")
        controls = fold.matched_warp_predictions
        if (any(type(value) is not MatchedWarpPrediction for value in controls)
                or tuple((value.variant, value.draw_index) for value in controls) != keys
                or any(type(value.draw_index) is not int for value in controls)):
            raise ValueError("all eight matched-warp draws for every variant are required in fixed order")
    weights = condition_weights()
    fitted = np.empty((len(expected), 4, 2), dtype=np.float64)
    controls = np.empty((len(expected), 4, MATCHED_WARP_DRAWS, 2), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for person, fold in enumerate(run.folds):
            gain = np.asarray(fold.gain)
            if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
                raise ValueError("one positive finite common Task-A gain is required")
            G = _array(fold.G, (18, support_size), "fold source maps")
            observed = _array(fold.observed_partitions, (4, 18, support_size), "observed recipient B partitions")
            halves = (observed[(0, 2), :, :]+observed[(1, 3), :, :])/2
            residual = halves-G
            residual -= residual.mean(axis=-1, keepdims=True)

            def alignment(prediction):
                pattern = _array(prediction, (18, support_size), "M2 source-map prediction")/float(gain)-G
                pattern -= pattern.mean(axis=-1, keepdims=True)
                return np.sum(np.mean(residual*pattern[None], axis=-1)*weights, axis=-1)

            for variant in range(4):
                fitted[person, variant] = alignment(fold.m2_predictions[variant].maps)
                for draw in range(MATCHED_WARP_DRAWS):
                    controls[person, variant, draw] = alignment(
                        fold.matched_warp_predictions[variant*MATCHED_WARP_DRAWS+draw].maps)
        advantage = fitted-controls.mean(axis=2)
    if not all(np.isfinite(value).all() for value in (fitted, controls, advantage)):
        raise ArithmeticError("nonfinite alignment; no dropping, clipping or substitution")
    variants = {}
    for index, name in enumerate(M2_VARIANTS):
        participant_values = advantage[:, index].mean(axis=1)
        variants[name] = {
            "fitted_alignment_by_participant_and_half": _array(
                fitted[:, index], (len(expected), 2), "fitted alignment"),
            "control_alignment_by_participant_draw_and_half": _array(
                controls[:, index], (len(expected), MATCHED_WARP_DRAWS, 2), "control alignments"),
            "advantage_by_participant_and_half": _array(
                advantage[:, index], (len(expected), 2), "half alignment advantages"),
            "participant_alignment_advantage": _array(
                participant_values, (len(expected),), "participant alignment advantages"),
            "mean_fitted_alignment": float(fitted[:, index].mean()),
            "mean_control_alignment": float(controls[:, index].mean()),
            "alignment_advantage": float(participant_values.mean()),
        }
    return {
        "participant_ids": expected, "deleted_id": run.deleted_id,
        "biological_n": len(expected), "matched_draws": MATCHED_WARP_DRAWS,
        "support_size": support_size, "voxel_weight": 1 / support_size,
        "field_pattern": "fitted_prediction_divided_by_common_Task_A_gain_minus_fold_G",
        "observed_residual_baseline": "fold_G",
        "statistic": "equal_participant_mean_fitted_minus_mean_matched_warp_alignment",
        "variants": variants,
        "half_field_stability_certified": False, "native_signature_certified": False,
        "permutation_inference": False,
    }
