"""Decompose the completed repeated-image control's unchanged residual error.

This is an outcome-informed diagnostic, not a corrected score or model search.
The two original outer fits are repeated at their already selected penalties.
Only after their losses reproduce the released control do we describe how much
held-out, training-whitened error lies in each sequence's degree-0..3 time
polynomial subspace. The projection uses held-out residuals descriptively; it
does not alter predictions, fitting, whitening, tuning, or the primary R.
"""
from __future__ import annotations

import numpy as np

import repeated_image_control as control
from continuous_sequence_model import JointSequenceDesign
from continuous_tuning import RIDGE_GRID
from core_reference_direct_ridge import DirectCoreRidge
from training_transforms import CommonSensorSpace

SCHEMA = "ep18_repeated_signal_diagnostic_v1"
MATCH_RTOL = 1e-9
MATCH_ATOL = 1e-12
DIAGNOSTIC_RECIPE = {
    "models": "unchanged_nuisance_and_200_identity_control",
    "selection": "reuse_each_completed_outer_direction_selected_penalty_no_retuning",
    "whitening": "refit_original_nuisance_residual_whitener_on_outer_training_only",
    "projection": "orthogonal_degree_0_to_3_time_Legendre_on_each_retained_sequence",
    "energy": "mean_squared_whitened_residual_over_retained_times_and_61_contrasts",
    "direction_denominator": "equal_sequence_mean_original_total_nuisance_MSE",
    "participant_aggregation": "equal_mean_of_two_direction_contributions",
    "participant_polynomial_fraction": "equal_mean_of_direction_ratios_of_sequence_mean_energies",
    "heldout_projection_role": "descriptive_error_decomposition_only",
    "reproduction_relative_tolerance": MATCH_RTOL,
    "reproduction_absolute_tolerance": MATCH_ATOL,
    "primary_R_changed": False,
}


def _match(actual, expected, reason):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    control.require(actual.shape == expected.shape and np.isfinite(actual).all()
                    and np.isfinite(expected).all()
                    and np.allclose(actual, expected, rtol=MATCH_RTOL, atol=MATCH_ATOL), reason)


def _validate_completed(person, sequences, recipe, completed):
    binding = {"schema": control.SCHEMA, "participant_id": person.participant_id,
        "data_role": "development", "recipe": recipe, "identity_recipe": control.IDENTITY_RECIPE,
        "image_ids": list(person.image_ids),
        "sequence_support": [{"sequence_id": item.sequence_id, "ordinal": int(item.ordinal),
            "samples": len(item.sample_times_seconds),
            "retained_samples": int(np.count_nonzero(item.retained_mask))} for item in sequences],
        "channel_order": list(sequences[0].channel_order)}
    control.require(completed.get("schema") == control.SCHEMA
        and completed.get("status") == "complete_repeated_image_control"
        and completed.get("participant_id") == person.participant_id
        and completed.get("data_role") == "development"
        and completed.get("binding") == binding
        and completed.get("image_identity_count") == 200
        and completed.get("physical_sequence_count") == 12
        and completed.get("audit_access") is False
        and completed.get("concept_or_false_models_computed") is False,
        "matching_complete_released_repeated_control_required")
    records = completed.get("directions", [])
    control.require(len(records) == 2
        and tuple(record.get("direction") for record in records) == control.DIRECTIONS,
        "both_completed_original_control_directions_required")
    for parity, record in enumerate(records):
        train = [item.sequence_id for item in sequences if item.ordinal % 2 == parity]
        test = [item.sequence_id for item in sequences if item.ordinal % 2 != parity]
        index = record.get("selected_penalty_index")
        means = np.asarray(record.get("inner_mean_normalized_identity_losses"), dtype=float)
        control.require(record.get("status") == "complete_direction"
            and record.get("binding") == binding and record.get("training_sequence_ids") == train
            and record.get("heldout_sequence_ids") == test
            and record.get("inner_validations_completed") == 6
            and means.shape == (13,) and np.isfinite(means).all() and np.all(means >= 0)
            and isinstance(index, int) and 0 <= index < 13
            and record.get("selected_ridge_penalty") == RIDGE_GRID[index],
            "unchanged_completed_outer_split_and_saved_selection_required")
        control.require(index == min(range(13), key=lambda i: (means[i], -RIDGE_GRID[i], i)),
                        "saved_control_selection_does_not_match_completed_inner_curve")
        raw, nuisance = (np.asarray(record.get(name), dtype=float) for name in
                         ("identity_sequence_losses", "nuisance_sequence_losses"))
        control.require(raw.shape == nuisance.shape == (6,), "all6_original_sequence_losses_required")
        _match(record.get("gain"), 1 - control._normal(raw, nuisance), "released_direction_gain_differs")
        whitening = record.get("training_whitening", {})
        control.require(whitening.get("sequence_ids") == train
                        and whitening.get("shrinkage_strength") == recipe["shrinkage_strength"],
                        "original_training_only_whitening_required")
    _match(completed.get("gain"), np.mean([record["gain"] for record in records]),
           "released_complete_participant_gain_differs")
    return records


def _polynomial_basis(times):
    times = np.asarray(times, dtype=float)
    control.require(times.ndim == 1 and len(times) >= 4 and np.isfinite(times).all()
                    and np.all(np.diff(times) > 0), "four_distinct_retained_times_required")
    # An affine change of the original E3-based time coordinate leaves this
    # degree-0..3 subspace unchanged; only its numerical conditioning changes.
    coordinate = 2 * (times - times[0]) / (times[-1] - times[0]) - 1
    basis, triangular = np.linalg.qr(np.polynomial.legendre.legvander(coordinate, 3), mode="reduced")
    control.require(np.all(np.abs(np.diag(triangular)) > np.finfo(float).eps * len(times)),
                    "full_rank_time_polynomial_projection_required")
    return basis


def _energies(residual, basis):
    residual = np.asarray(residual, dtype=float)
    control.require(residual.ndim == 2 and residual.shape[0] == basis.shape[0]
                    and residual.shape[1] > 0 and np.isfinite(residual).all(),
                    "finite_retained_whitened_residual_required")
    projected = basis @ (basis.T @ residual)
    values = {"total": float(np.mean(residual**2)),
              "polynomial": float(np.mean(projected**2)),
              "orthogonal": float(np.mean((residual - projected)**2))}
    control.require(all(np.isfinite(value) and value >= 0 for value in values.values()),
                    "finite_nonnegative_residual_energies_required")
    _match(values["polynomial"] + values["orthogonal"], values["total"],
           "orthogonal_decomposition_does_not_conserve_error")
    return values


def _summarize_direction(rows, original):
    means = {name: {part: float(np.mean([row[name][part] for row in rows]))
                   for part in ("total", "polynomial", "orthogonal")}
             for name in ("nuisance", "identity")}
    denominator = means["nuisance"]["total"]
    control.require(np.isfinite(denominator) and denominator > 0,
                    "positive_original_total_nuisance_denominator_required")
    for name in ("nuisance", "identity"):
        _match([row[name]["total"] for row in rows], original[name + "_sequence_losses"],
               "refitted_" + name + "_losses_do_not_reproduce_released_control")
    gain = float(1 - means["identity"]["total"] / denominator)
    # Ratios near one can subtract to an extremely small R. Compare those
    # ratios at the same tolerance rather than requiring relative accuracy in
    # the near-zero difference, after checking every original sequence loss.
    _match(1 - gain, 1 - original["gain"], "refitted_direction_R_does_not_reproduce_control")
    polynomial = (means["nuisance"]["polynomial"] - means["identity"]["polynomial"]) / denominator
    orthogonal = (means["nuisance"]["orthogonal"] - means["identity"]["orthogonal"]) / denominator
    _match(1 - (polynomial + orthogonal), 1 - gain, "component_contributions_do_not_add_to_R")
    return {"direction": original["direction"],
        "training_sequence_ids": original["training_sequence_ids"],
        "heldout_sequence_ids": original["heldout_sequence_ids"],
        "selected_penalty_index": original["selected_penalty_index"],
        "selected_ridge_penalty": original["selected_ridge_penalty"],
        "gain_original_R": float(original["gain"]), "gain_recomputed_R": gain,
        "polynomial_gain_contribution": float(polynomial),
        "orthogonal_gain_contribution": float(orthogonal),
        "nuisance_polynomial_energy_fraction": means["nuisance"]["polynomial"] / denominator,
        "mean_sequence_energies": means, "original_total_nuisance_denominator": denominator,
        "sequence_rows": rows, "released_losses_reproduced": True}


def run_participant(person, *, recipe, completed_control, budget_check, progress):
    """Return a complete two-direction diagnostic; never change or retune R."""
    recipe = control.validate_recipe(recipe)
    control.require(callable(budget_check) and callable(progress), "explicit_budget_and_progress_callbacks_required")
    sequences = control._roster(person, budget_check)
    originals = _validate_completed(person, sequences, recipe, completed_control)
    sensor = CommonSensorSpace(sequences[0].channel_order)
    designs = {}
    for item in sequences:
        budget_check()
        designs[item.sequence_id] = control._sequence_designs(item)
    backend = DirectCoreRidge(check=budget_check)
    results = []
    try:
        for parity, original in enumerate(originals):
            training = tuple(item for item in sequences if item.ordinal % 2 == parity)
            heldout = tuple(item for item in sequences if item.ordinal % 2 != parity)
            progress(original["direction"] + ":refit_saved_outer_penalty")
            state = control._prepare_training(training, designs, sensor, recipe, backend, budget_check)
            fit = control._identity_fit(state, original["selected_penalty_index"], recipe, backend, budget_check)
            trace = float(np.trace(state.whitener.regularized_covariance))
            _match(trace, original["training_whitening"]["regularized_covariance_trace"],
                   "refitted_training_whitening_does_not_reproduce_control")
            rows = []
            for item in heldout:
                budget_check()
                mask = item.retained_mask
                target = sensor.apply(item.sensor_samples, channel_order=item.channel_order)
                basis = _polynomial_basis(np.asarray(item.sample_times_seconds)[mask])
                row = {"sequence_id": item.sequence_id, "retained_samples": int(np.count_nonzero(mask))}
                for name, coefficients in (("nuisance", state.nuisance_fit.coefficients),
                                           ("identity", fit.coefficients)):
                    prediction = backend.predict(JointSequenceDesign([designs[item.sequence_id][name]]), coefficients)
                    residual = state.whitener.apply(target - prediction)[mask]
                    row[name] = _energies(residual, basis)
                rows.append(row)
            result = _summarize_direction(rows, original)
            result["training_whitening"] = {"sequence_ids": list(state.sequence_ids),
                "shrinkage_strength": state.whitener.shrinkage_strength,
                "regularized_covariance_trace": trace}
            results.append(result)
            del state, fit
            backend.clear()
            progress(original["direction"] + ":decomposition_complete")
    finally:
        backend.clear()
    control.require(tuple(result["direction"] for result in results) == control.DIRECTIONS,
                    "both_directions_required_for_complete_diagnostic")
    result = {"schema": SCHEMA, "status": "complete_repeated_signal_diagnostic",
        "participant_id": person.participant_id, "data_role": "development",
        "diagnostic_recipe": DIAGNOSTIC_RECIPE, "source_control_schema": control.SCHEMA,
        "image_identity_count": 200, "physical_sequence_count": 12,
        "directions": results, "inference_unit": "participant", "cohort_inference_performed": False,
        "primary_R_changed": False, "inner_retuning_performed": False,
        "concept_or_false_models_computed": False, "audit_access": False}
    for key in ("gain_original_R", "gain_recomputed_R", "polynomial_gain_contribution",
                "orthogonal_gain_contribution", "nuisance_polynomial_energy_fraction"):
        result[key] = float(np.mean([direction[key] for direction in results]))
    result["mean_sequence_energies"] = {
        name: {part: float(np.mean([direction["mean_sequence_energies"][name][part]
                                  for direction in results]))
               for part in ("total", "polynomial", "orthogonal")}
        for name in ("nuisance", "identity")}
    _match(1 - result["gain_recomputed_R"], 1 - completed_control["gain"],
           "refitted_complete_participant_R_does_not_reproduce_control")
    budget_check()
    progress("participant_diagnostic_complete")
    return result
