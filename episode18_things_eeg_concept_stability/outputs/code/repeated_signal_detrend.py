"""One fixed degree-three sequence residualization for signal diagnosis only.

This outcome-informed follow-up changes a copied diagnostic input, not the
original EEG, registered preprocessing, models, R or pass rules. Each of the
62 raw channels is projected independently on the same retained-time degree
0..3 polynomial subspace. Whole-sequence projection uses all retained samples;
the resulting template similarity is descriptive, not held-out prediction.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from repeated_image_control import _roster, require
from repeated_image_sequences import RepeatedParticipant, RepeatedSequence
from repeated_signal_diagnostic import _polynomial_basis
from repeated_signal_structure import measure_structure


SCHEMA = "ep18_repeated_signal_detrend_v1"
PROJECTION_RECIPE = {
    "choice_count": 1,
    "degree": 3,
    "basis": "degree_0_to_3_time_Legendre_QR_on_original_retained_samples",
    "scope": "independent_physical_sequence_and_each_of_62_original_channels",
    "retained_rows": "unchanged_original_scoring_support",
    "nonretained_rows": "unchanged_copied_values",
    "event_times_and_image_order": "unchanged_original_acquisition_ticks",
    "structure": "unchanged_fixed_61_contrasts_200_images_40_lag_bins_and_shift37",
    "lag_windows_relative_to_original_event_ms": [[0, 800], [-800, 0]],
    "negative_lag_control": "one_fixed_descriptive_window_not_registered_control_or_a_search",
    "window_endpoints": "both_closed_at_their_final_endpoint_shared_observed_0ms_sample_when_present",
    "energy_units": "V_squared_per_channel_then_summed_over_62_channels",
    "participant_energy_aggregation": "equal_mean_of_12_sequence_energies",
    "fraction_aggregation": "equal_mean_of_12_sequence_fractions_without_exclusion",
    "whole_sequence_data_used": True,
    "out_of_sample_prediction": False,
    "original_R_refit_or_rescored": False,
    "preprocessing_search": False,
}


def _readonly_copy(value):
    copied = np.array(value, copy=True)
    copied.flags.writeable = False
    return copied


def _project_sequence(item, check):
    check()
    mask = np.asarray(item.retained_mask)
    raw = np.asarray(item.sensor_samples, dtype=float)
    retained = raw[mask]
    basis = _polynomial_basis(np.asarray(item.sample_times_seconds)[mask])
    coefficients = basis.T @ retained
    removed = basis @ coefficients
    residual = retained - removed
    total = np.mean(retained**2, axis=0)
    polynomial = np.mean(removed**2, axis=0)
    orthogonal = np.mean(residual**2, axis=0)
    require(all(value.shape == (62,) and np.isfinite(value).all() and np.all(value >= 0)
                for value in (total, polynomial, orthogonal)), "finite_complete_62channel_projection_energies_required")
    scale = np.maximum(total, np.finfo(float).tiny)
    require(np.all(np.abs(polynomial + orthogonal - total) <= 1e-10 * scale),
            "raw_projection_must_conserve_each_channel_energy")
    norm = float(np.linalg.norm(retained))
    orthogonality = float(np.linalg.norm(basis.T @ residual) / max(norm, np.finfo(float).tiny))
    require(np.isfinite(orthogonality) and orthogonality <= 1e-10,
            "raw_residual_must_be_orthogonal_to_fixed_polynomial_basis")
    changed = raw.copy()
    changed[mask] = residual
    changed.flags.writeable = False
    # Copy every array: the diagnostic can neither mutate the source nor
    # receive later mutations through a shared writable fixture/input array.
    copied = replace(item, sensor_samples=changed,
        sample_times_seconds=_readonly_copy(item.sample_times_seconds),
        retained_mask=_readonly_copy(item.retained_mask),
        event_onsets_seconds=_readonly_copy(item.event_onsets_seconds),
        image_indices=_readonly_copy(item.image_indices),
        target_flags=_readonly_copy(item.target_flags), response_flags=_readonly_copy(item.response_flags))
    total_sum, removed_sum, residual_sum = (float(value.sum()) for value in (total, polynomial, orthogonal))
    row = {"sequence_id": item.sequence_id, "ordinal": int(item.ordinal),
        "retained_samples": int(mask.sum()), "sample_count": len(mask),
        "per_channel_total_energy_V2": total.tolist(),
        "per_channel_polynomial_energy_V2": polynomial.tolist(),
        "per_channel_residual_energy_V2": orthogonal.tolist(),
        "total_energy_V2": total_sum, "polynomial_energy_V2": removed_sum,
        "residual_energy_V2": residual_sum,
        "removed_energy_fraction": removed_sum / total_sum if total_sum > 0 else None,
        "energy_conservation_difference_V2": removed_sum + residual_sum - total_sum,
        "polynomial_residual_orthogonality_norm_ratio": orthogonality,
        "nonretained_samples_unchanged": True}
    check()
    return copied, row


def _project_person(person, *, budget_check):
    require(type(person) is RepeatedParticipant
            and all(type(item) is RepeatedSequence for item in person.sequences),
            "unchanged_repeated_reader_dataclasses_required")
    sequences = _roster(person, budget_check)
    changed, rows = [], []
    for item in sequences:
        copied, accounting = _project_sequence(item, budget_check)
        changed.append(copied)
        rows.append(accounting)
    summary = deepcopy(person.metadata_summary)
    summary["diagnostic_only_fixed_polynomial_residualization"] = deepcopy(PROJECTION_RECIPE)
    copied_person = replace(person, sequences=tuple(changed), metadata_summary=summary)
    means = {key: float(np.mean([row[key] for row in rows]))
             for key in ("total_energy_V2", "polynomial_energy_V2", "residual_energy_V2")}
    fractions = [row["removed_energy_fraction"] for row in rows]
    accounting = {"sequence_rows": rows, "channel_order": list(sequences[0].channel_order),
        "mean_sequence_energies": means,
        "mean_sequence_removed_energy_fraction": None if any(value is None for value in fractions)
                                                   else float(np.mean(fractions)),
        "all_12_sequences_accounted": True, "all_62_channels_accounted": True,
        "energy_space": "raw_62_channel_voltage_not_training_whitened_61_contrast_error"}
    return copied_person, accounting


def run_participant(person, *, budget_check):
    require(callable(budget_check), "explicit_resource_callback_required")
    copied, accounting = _project_person(person, budget_check=budget_check)
    structure = measure_structure(copied, budget_check=budget_check)
    # The shared function performs no detrending itself; disclose the explicit
    # diagnostic preprocessing of its copied input in this returned record.
    structure.update(raw_EEG_detrending=True,
        input_preprocessing="fixed_degree3_whole_sequence_projection_on_original_retained_samples_only",
        diagnostic_only_preprocessing=True, out_of_sample_prediction=False,
        original_R_changed=False, original_control_threshold_changed=False, primary_rescue=False)
    structure["template_definition"].update(lag_window_relative_to_original_event_ms=[0, 800],
        diagnostic_onset_shift_ms=0,
        boundary_overlap="positive and negative windows both include an observed original0ms sample when present")
    # Construct a separate diagnostic view. No acquisition event is rounded to
    # the 250-Hz grid or changed in the authentic reader/person. Shifting the
    # copied final-offset marker preserves the shared dataclass consistency
    # check; the structure diagnostic does not fit nuisance/event models.
    shifted = []
    for item in copied.sequences:
        budget_check()
        ticks = np.rint(np.asarray(item.event_onsets_seconds) * 1000).astype(np.int64)
        require(np.all(np.abs(ticks / 1000. - item.event_onsets_seconds) < 1e-10),
                "negative_window_requires_exact_original_acquisition_ticks")
        onsets = _readonly_copy((ticks - 800) / 1000.)
        shifted.append(replace(item, event_onsets_seconds=onsets,
                               final_offset_seconds=float(onsets[-1] + .05)))
    negative = measure_structure(replace(copied, sequences=tuple(shifted)), budget_check=budget_check)
    negative.update(raw_EEG_detrending=True,
        input_preprocessing="fixed_degree3_whole_sequence_projection_on_original_retained_samples_only",
        diagnostic_only_preprocessing=True, out_of_sample_prediction=False,
        original_R_changed=False, original_control_threshold_changed=False, primary_rescue=False,
        registered_negative_lag_control_score=False)
    negative["template_definition"].update(lag_bin_edges_ms=list(range(-800, 1, 20)),
        lag_window_relative_to_original_event_ms=[-800, 0], final_bin_includes_800ms=False,
        final_bin_includes_0ms=True, diagnostic_onset_shift_ms=-800,
        boundary_overlap="positive and negative windows both include an observed original0ms sample when present")
    budget_check()
    return {"schema": SCHEMA, "status": "complete_repeated_signal_detrend",
        "participant_id": person.participant_id,
        "data_role": "development_repeated_signal_recovery_diagnostic_only",
        "projection_recipe": deepcopy(PROJECTION_RECIPE), "projection_accounting": accounting,
        "structure": structure, "negative_lag_structure": negative,
        "physical_sequence_count": 12, "image_identity_count": 200,
        "recorded_channel_count": 62, "source_arrays_changed": False,
        "nonretained_samples_changed": False, "original_R_changed": False,
        "original_R_refit_or_rescored": False, "original_control_threshold_changed": False,
        "primary_rescue": False, "out_of_sample_prediction": False,
        "model_fit_performed": False, "inference_performed": False,
        "audit_access": False, "concept_or_false_scores_computed": False,
        "interpretation": "Fixed post hoc diagnostic of whole-sequence slow structure; no unchanged-control pass, held-out prediction or signal-absence claim."}
