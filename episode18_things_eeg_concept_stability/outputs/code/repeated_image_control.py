"""Prespecified repeated-image identity-template control on development EEG.

The caller supplies the permitted, independently filtered physical sequences.
Odd/even means chronological human sequence numbers, not image/event parity.
Each outer direction selects one of the original thirteen ridge penalties by
leave-one-training-sequence-out validation. Every such training set refits its
nuisance model and residual whitener; no outer evaluation EEG enters selection.
The 200 identity indicators are uncentered and unscaled; the shared evoked mean
is also represented by the unchanged nuisance design. Both models fit EEG
jointly in the fixed 61-contrast sensor space. This module makes no cohort or
concept claim and never reads image features, concept labels or audit records.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from continuous_fold_runner import SolverLimits
from continuous_sequence_model import JointSequenceDesign, SequenceDesign, _PreparedTerm
from continuous_tuning import RIDGE_GRID, normalize_losses
from core_reference_direct_ridge import DirectCoreRidge
from production_continuous_designs import TemporalBasis, _compose, _expanded, _kernels, readonly
from production_core_reference import validate_recipe as validate_core_recipe
from training_transforms import CommonSensorSpace, fit_residual_whitener

SCHEMA = "ep18_repeated_image_control_v1"
DIRECTIONS = ("odd_to_even", "even_to_odd")
IDENTITY_RECIPE = {
    "identity_encoding": "one_hot_uncentered_200",
    "temporal_basis": "raised_cosine_12",
    "outer_folds": "chronological_physical_sequences_odd_to_even_and_even_to_odd",
    "inner_folds": "leave_one_training_physical_sequence_out",
    "inner_selection": "mean_inner_ratio_of_equal_sequence_means_then_larger_lambda",
    "ridge_grid": [float(value) for value in RIDGE_GRID],
    "direction_gain": "1_minus_ratio_of_equal_sequence_mean_whitened_MSEs",
    "participant_gain": "equal_mean_of_both_direction_gains",
    "drift_span_seconds": "last_event_onset_plus_0.801_minus_E3_cue",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_recipe(recipe):
    """Accept the existing complete core recipe, not a second tuning contract."""
    value = validate_core_recipe(recipe)
    require(value["intercept_penalty"] == 0 and value["nuisance_penalty"] == 1e-6
            and value["shrinkage_strength"] == .1
            and value["solver"] == {"atol": 1e-8, "btol": 1e-8,
                                    "conlim": 1e12, "maxiter": 10000},
            "unchanged_core_numerical_recipe_required")
    return value


def _json_write(path, record):
    data = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    require(len(data.encode()) <= 16 * 1024**2, "bounded_control_checkpoint_required")
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _json_read(path):
    require(path.stat().st_size <= 16 * 1024**2, "bounded_control_checkpoint_required")
    with path.open() as source:
        return json.load(source)


def _roster(person, check):
    sequences = tuple(sorted(person.sequences, key=lambda item: item.ordinal))
    require(len(person.image_ids) == len(set(person.image_ids)) == 200
            and len(sequences) == 12
            and tuple(item.ordinal for item in sequences) == tuple(range(12))
            and len({item.sequence_id for item in sequences}) == 12,
            "all200_identities_and12_chronological_physical_sequences_required")
    channels = tuple(sequences[0].channel_order)
    require(len(channels) == len(set(channels)) == 62, "fixed62_channel_order_required")
    for item in sequences:
        check()
        times, samples, mask = (np.asarray(item.sample_times_seconds),
                               np.asarray(item.sensor_samples), np.asarray(item.retained_mask))
        onsets, indices = np.asarray(item.event_onsets_seconds), np.asarray(item.image_indices)
        require(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all()
                and np.all(np.diff(times) > 0) and samples.shape == (len(times), 62)
                and np.isfinite(samples).all() and tuple(item.channel_order) == channels
                and mask.dtype == np.bool_ and mask.shape == times.shape and mask.any(),
                "finite_fixed_sensor_samples_and_model_independent_retention_required")
        require(onsets.shape == (200,) and np.isfinite(onsets).all()
                and np.all(np.diff(onsets) > 0) and indices.dtype.kind in "iu"
                and np.array_equal(np.sort(indices), np.arange(200)),
                "each_physical_sequence_requires_each_of200_identities_once")
        require(np.isfinite(item.cue_seconds) and item.cue_seconds == 0.
                and np.isfinite(item.final_offset_seconds)
                and abs(item.final_offset_seconds - onsets[-1] - .05) < 1e-10
                and onsets[0] > item.cue_seconds,
                "actual_E3_and_fixed_final_stimulus_offset_required")
        for flags in (item.target_flags, item.response_flags):
            values = np.asarray(flags)
            require(values.shape == (200,) and np.isin(values, (0, 1)).all(),
                    "original_binary_target_and_coarse_response_flags_required")
    return sequences


def _sequence_designs(item):
    """Reuse the original 88 nuisance columns, with 200 presentation positions."""
    times = np.asarray(item.sample_times_seconds)
    onsets = np.asarray(item.event_onsets_seconds)
    basis = TemporalBasis("raised_cosine_12")
    ticks, event_ticks = np.rint(times * 1000).astype(np.int64), np.rint(onsets * 1000).astype(np.int64)
    require(np.all(np.abs(ticks / 1000. - times) < 1e-10)
            and np.all(np.abs(event_ticks / 1000. - onsets) < 1e-10),
            "unaltered_acquisition_tick_times_required")
    kernels = _kernels(ticks, event_ticks, basis)
    boundaries = _kernels(ticks, np.rint(np.array([item.cue_seconds,
                            item.final_offset_seconds]) * 1000).astype(np.int64), basis)
    duration = float(onsets[-1] + .801 - item.cue_seconds)
    x = 2 * (times - item.cue_seconds) / duration - 1
    drift = np.column_stack((np.ones(len(x)), x, .5 * (3*x*x - 1), .5 * (5*x*x*x - 3*x)))
    position = 2 * np.arange(200) / 199. - 1
    positions = np.column_stack((np.ones(200), position, .5 * (3*position*position - 1)))
    flags = np.column_stack((item.target_flags, item.response_flags))
    nuisance = np.column_stack((drift, _expanded(kernels, positions, len(times), 12),
        _expanded(boundaries, np.eye(2), len(times), 12), _expanded(kernels, flags, len(times), 12)))
    identity = readonly(np.eye(200)[np.asarray(item.image_indices)])
    reference = SequenceDesign(item.sequence_id, times, nuisance, {})
    # Reuse the exact acquisition-tick kernels in both models, as in the core
    # builder. Subtracting floating absolute times could lose a closed 800-ms
    # endpoint; no event time is rounded to the lower-rate sampling grid.
    term = _PreparedTerm(identity, kernels, 12, slice(0, 0))
    designs = {
        "nuisance": reference,
        "identity": _compose(reference, reference.nuisance, {"identity": term}, basis),
    }
    require(designs["nuisance"].shape[1] == 88 and designs["identity"].shape[1] == 2488,
            "unchanged88_nuisance_plus2400_identity_coefficients_required")
    return designs


@dataclass
class TrainingState:
    sequence_ids: tuple
    designs: dict
    targets: np.ndarray
    weights: np.ndarray
    nuisance_penalty: np.ndarray
    nuisance_fit: object
    whitener: object


def _prepare_training(items, designs, sensor, recipe, backend, check):
    check()
    train = {name: JointSequenceDesign([designs[item.sequence_id][name] for item in items])
             for name in ("nuisance", "identity")}
    targets = np.concatenate([sensor.apply(item.sensor_samples, channel_order=item.channel_order)
                              for item in items])
    weights = np.concatenate([item.retained_mask / (len(items) * np.count_nonzero(item.retained_mask))
                              for item in items])
    penalties = np.full(train["nuisance"].shape[1], recipe["nuisance_penalty"])
    penalties[0] = recipe["intercept_penalty"]
    fit = backend.fit(train["nuisance"], targets, weights, penalties,
                      SolverLimits(**recipe["solver"]), check)
    residual = (targets - backend.predict(train["nuisance"], fit.coefficients))[weights > 0]
    covariance_weights = weights[weights > 0]
    centered = residual - covariance_weights @ residual / covariance_weights.sum()
    trace = float(np.sum(covariance_weights[:, None] * centered**2) / covariance_weights.sum())
    whitener = fit_residual_whitener(residual, sample_weights=covariance_weights,
        center_covariance=True, covariance_denominator=float(covariance_weights.sum()),
        shrinkage_target=np.eye(61) * (trace / 61.), shrinkage_strength=recipe["shrinkage_strength"])
    return TrainingState(tuple(item.sequence_id for item in items), train, targets, weights,
                         penalties, fit, whitener)


def _score(items, designs, sensor, state, model, coefficients, backend, check):
    errors = []
    for item in items:
        check()
        target = sensor.apply(item.sensor_samples, channel_order=item.channel_order)
        prediction = backend.predict(JointSequenceDesign([designs[item.sequence_id][model]]), coefficients)
        residual = state.whitener.apply(target - prediction)[item.retained_mask]
        error = float(np.mean(residual**2))
        require(np.isfinite(error) and error >= 0, "finite_nonnegative_sequence_MSE_required")
        errors.append(error)
    return np.asarray(errors)


def _identity_fit(state, index, recipe, backend, check):
    penalty = np.concatenate((state.nuisance_penalty,
        np.full(state.designs["identity"].shape[1] - len(state.nuisance_penalty), RIDGE_GRID[index])))
    return backend.fit(state.designs["identity"], state.targets, state.weights, penalty,
                       SolverLimits(**recipe["solver"]), check)


def _normal(raw, nuisance):
    return normalize_losses(raw, nuisance, normalization="ratio_of_equal_sequence_means")


def _load_inner(path, binding, train_ids, heldout_id):
    record = _json_read(path)
    require(record.get("status") == "complete_inner" and record.get("binding") == binding
            and record.get("training_sequence_ids") == list(train_ids)
            and record.get("heldout_sequence_ids") == [heldout_id],
            "complete_matching_inner_checkpoint_required")
    raw, nuisance = np.asarray(record["identity_sequence_losses"]), np.asarray(record["nuisance_sequence_losses"])
    require(raw.shape == (13, 1) and nuisance.shape == (1,), "complete13_penalty_inner_checkpoint_required")
    expected = _normal(raw, nuisance)
    require(np.array_equal(np.asarray(record["normalized_identity_losses"]), expected),
            "inner_checkpoint_normalization_differs")
    return record


def _load_direction(path, binding, direction, train_ids, test_ids):
    record = _json_read(path)
    require(record.get("status") == "complete_direction" and record.get("binding") == binding
            and record.get("direction") == direction
            and record.get("training_sequence_ids") == list(train_ids)
            and record.get("heldout_sequence_ids") == list(test_ids),
            "complete_matching_direction_checkpoint_required")
    raw, nuisance = np.asarray(record["identity_sequence_losses"]), np.asarray(record["nuisance_sequence_losses"])
    require(raw.shape == nuisance.shape == (6,), "all6_heldout_sequences_required")
    gain = float(1 - _normal(raw, nuisance))
    means = np.asarray(record["inner_mean_normalized_identity_losses"])
    require(means.shape == (13,) and np.isfinite(means).all() and np.all(means >= 0)
            and record.get("inner_validations_completed") == 6, "all6_inner_validations_required")
    choice = min(range(13), key=lambda index: (means[index], -RIDGE_GRID[index], index))
    require(record.get("selected_penalty_index") == choice and record.get("selected_ridge_penalty") == RIDGE_GRID[choice]
            and record.get("gain") == gain, "direction_selection_or_gain_differs")
    whitening = record.get("training_whitening", {})
    require(whitening.get("sequence_ids") == list(train_ids)
            and whitening.get("shrinkage_strength") == .1
            and isinstance(whitening.get("regularized_covariance_trace"), (int, float))
            and np.isfinite(whitening["regularized_covariance_trace"])
            and whitening["regularized_covariance_trace"] > 0,
            "complete_training_only_whitening_checkpoint_required")
    return record


def _fit_direction(training, heldout, *, direction, designs, sensor, recipe,
                   directory, binding, backend, check, progress):
    inner = []
    for omitted in training:
        check()
        subset = tuple(item for item in training if item.sequence_id != omitted.sequence_id)
        path = directory / (direction + "_inner_%02d.json" % omitted.ordinal)
        if path.exists():
            record = _load_inner(path, binding, tuple(item.sequence_id for item in subset), omitted.sequence_id)
        else:
            progress(direction + ":inner_%02d_training" % omitted.ordinal)
            state = _prepare_training(subset, designs, sensor, recipe, backend, check)
            nuisance = _score((omitted,), designs, sensor, state, "nuisance",
                              state.nuisance_fit.coefficients, backend, check)
            raw = []
            for index in range(13):
                check()
                progress(direction + ":inner_%02d_penalty_%02d" % (omitted.ordinal, index))
                fit = _identity_fit(state, index, recipe, backend, check)
                raw.append(_score((omitted,), designs, sensor, state, "identity", fit.coefficients, backend, check))
            raw = np.asarray(raw)
            record = {"status": "complete_inner", "binding": binding,
                "training_sequence_ids": list(state.sequence_ids), "heldout_sequence_ids": [omitted.sequence_id],
                "identity_sequence_losses": raw.tolist(), "nuisance_sequence_losses": nuisance.tolist(),
                "normalized_identity_losses": _normal(raw, nuisance).tolist()}
            _json_write(path, record)
            del state
            backend.clear()
        inner.append(record)
    means = np.mean([record["normalized_identity_losses"] for record in inner], axis=0)
    index = min(range(13), key=lambda choice: (means[choice], -RIDGE_GRID[choice], choice))
    progress(direction + ":outer_training")
    state = _prepare_training(training, designs, sensor, recipe, backend, check)
    nuisance = _score(heldout, designs, sensor, state, "nuisance", state.nuisance_fit.coefficients, backend, check)
    fit = _identity_fit(state, index, recipe, backend, check)
    raw = _score(heldout, designs, sensor, state, "identity", fit.coefficients, backend, check)
    result = {"status": "complete_direction", "binding": binding, "direction": direction,
        "training_sequence_ids": list(state.sequence_ids), "heldout_sequence_ids": [item.sequence_id for item in heldout],
        "inner_validations_completed": len(inner), "inner_mean_normalized_identity_losses": means.tolist(),
        "selected_penalty_index": index, "selected_ridge_penalty": float(RIDGE_GRID[index]),
        "identity_sequence_losses": raw.tolist(), "nuisance_sequence_losses": nuisance.tolist(),
        "gain": float(1 - _normal(raw, nuisance)),
        "training_whitening": {"sequence_ids": list(state.sequence_ids),
            "shrinkage_strength": state.whitener.shrinkage_strength,
            "regularized_covariance_trace": float(np.trace(state.whitener.regularized_covariance))},
        "solver": dict(backend.stats)}
    backend.clear()
    check()
    return result


def run_participant(person, *, recipe, run_dir, budget_check, progress):
    """Complete/reuse both outer directions; never emit a partial-person gain."""
    recipe = validate_recipe(recipe)
    require(callable(budget_check) and callable(progress), "explicit_budget_and_progress_callbacks_required")
    sequences = _roster(person, budget_check)
    directory = Path(run_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    binding = {"schema": SCHEMA, "participant_id": person.participant_id, "data_role": "development",
        "recipe": recipe, "identity_recipe": IDENTITY_RECIPE, "image_ids": list(person.image_ids),
        "sequence_support": [{"sequence_id": item.sequence_id, "ordinal": int(item.ordinal),
            "samples": len(item.sample_times_seconds), "retained_samples": int(np.count_nonzero(item.retained_mask))}
            for item in sequences], "channel_order": list(sequences[0].channel_order)}
    sensor = CommonSensorSpace(sequences[0].channel_order)
    designs = {item.sequence_id: _sequence_designs(item) for item in sequences}
    backend = DirectCoreRidge(check=budget_check)
    results = []
    for parity, direction in enumerate(DIRECTIONS):
        training = tuple(item for item in sequences if item.ordinal % 2 == parity)
        heldout = tuple(item for item in sequences if item.ordinal % 2 != parity)
        path = directory / (direction + ".complete.json")
        if path.exists():
            record = _load_direction(path, binding, direction,
                tuple(item.sequence_id for item in training), tuple(item.sequence_id for item in heldout))
            progress(direction + ":reused_complete")
        else:
            record = _fit_direction(training, heldout, direction=direction, designs=designs,
                sensor=sensor, recipe=recipe, directory=directory, binding=binding, backend=backend,
                check=budget_check, progress=progress)
            _json_write(path, record)
        results.append(record)
    require(tuple(record["direction"] for record in results) == DIRECTIONS,
            "both_original_directions_required_for_participant_gain")
    result = {"schema": SCHEMA, "status": "complete_repeated_image_control",
        "participant_id": person.participant_id, "data_role": "development", "binding": binding,
        "gain": float(np.mean([record["gain"] for record in results])), "directions": results,
        "image_identity_count": 200, "physical_sequence_count": 12,
        "concept_or_false_models_computed": False, "audit_access": False,
        "inference_unit": "participant", "cohort_inference_performed": False}
    budget_check()
    complete_path = directory / "complete.json"
    if complete_path.exists():
        require(_json_read(complete_path) == result, "complete_participant_checkpoint_differs")
    else:
        _json_write(complete_path, result)
    progress("participant_complete")
    return result
