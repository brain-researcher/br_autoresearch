"""One frozen exploratory selector refit; original control files stay intact.

Only changed outer directions are fitted. Their selected penalties come from
the complete saved-inner comparison, never from outer EEG. Heavy model imports
are local so the reporting consumer can share the selection validator cheaply.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

import report_repeated_selector_sensitivity as selector

SCHEMA = "ep18_repeated_selector_refit_v1"
STATUS = "complete_repeated_selector_refit"
MATCH_RTOL, MATCH_ATOL = 1e-9, 1e-12
SELECTOR_PATH = selector.EPISODE / "outputs/runs/repeated_selector_job_47586225_step_2/report/repeated_selector_sensitivity.complete.json"
REFIT_ROOT = selector.EPISODE / "outputs/runs/repeated_selector_refit/pooled001"
require = selector.require


def _match(actual, expected, reason):
    first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(first.shape == second.shape and np.isfinite(first).all()
        and np.isfinite(second).all()
        and np.allclose(first, second, rtol=MATCH_RTOL, atol=MATCH_ATOL), reason)


def validate_selection_report(record, *, recipe):
    """Validate the already frozen full vector; never inspect any outer loss."""
    require(record.get("schema") == selector.SCHEMA
        and record.get("status") == "complete_frozen_repeated_selector_sensitivity"
        and record.get("participants") == list(selector.PEOPLE)
        and record.get("selection_recipe") == selector.SELECTION_RECIPE
        and record.get("participant_count") == 30 and record.get("direction_count") == 60
        and record.get("inner_checkpoint_count") == 360 and record.get("ridge_penalty_count") == 13
        and record.get("selection_vector_complete") is True
        and record.get("data_role") == "development_repeated_signal_recovery_only"
        and Path(record["input_root"]).resolve() == selector.INPUT_ROOT.resolve()
        and all(record.get(key) is False for key in ("outer_losses_read", "EEG_read",
            "outer_refits_performed", "original_R_changed", "primary_rescue", "audit_access",
            "concept_or_false_scores_computed")), "complete original frozen selector report required")
    participants = record["participant_records"]
    require([row.get("participant_id") for row in participants] == list(selector.PEOPLE),
        "all thirty original selector participant entries required")
    vector = []
    for person in participants:
        person_id = person["participant_id"]
        support = selector._binding(person["binding"], person_id, recipe)
        require([row.get("direction") for row in person["directions"]] == list(selector.DIRECTIONS),
            "both original selector directions required")
        for parity, row in enumerate(person["directions"]):
            training = [item for item in support if item["ordinal"] % 2 == parity]
            expected_folds = [{"checkpoint": f"{person_id}/{row['direction']}_inner_{item['ordinal']:02d}.json",
                "heldout_ordinal": item["ordinal"], "heldout_sequence_id": item["sequence_id"],
                "training_sequence_ids": [other["sequence_id"] for other in training if other != item]}
                for item in training]
            require(row.get("inner_validations_completed") == 6 and row.get("inner_folds") == expected_folds
                and row.get("outer_training_sequence_ids") == [item["sequence_id"] for item in training],
                "unchanged six physical-sequence inner folds required")
            raw = selector._numeric(row["identity_inner_losses_by_fold_and_penalty"], (6, 13))
            nuisance = selector._numeric(row["nuisance_inner_losses_by_fold"], (6,))
            denominator = float(nuisance.sum())
            require(np.all(nuisance > 0) and np.isfinite(denominator), "positive finite inner denominators required")
            old_objective = np.mean(raw / nuisance[:, None], axis=0)
            new_objective = np.sum(raw, axis=0) / denominator
            require(np.array_equal(row["original_mean_normalized_identity_losses"], old_objective)
                and np.array_equal(row["alternative_pooled_normalized_identity_losses"], new_objective)
                and np.array_equal(row["alternative_weights_on_inner_ratios"], nuisance / denominator),
                "frozen selector objectives and weights must reproduce their saved inner losses")
            old, new = selector._choice(old_objective), selector._choice(new_objective)
            expected = {"participant_id": person_id, "direction": row["direction"],
                "original_selected_penalty_index": old, "original_selected_ridge_penalty": selector.RIDGE_GRID[old],
                "alternative_selected_penalty_index": new, "alternative_selected_ridge_penalty": selector.RIDGE_GRID[new],
                "selection_changed": old != new}
            require(all(row.get(key) == value for key, value in expected.items() if key != "participant_id"),
                "fixed exact larger-penalty selector choice required")
            vector.append(expected)
    changed = [row for row in vector if row["selection_changed"]]
    require(record["selection_vector"] == vector and record["changed_directions"] == changed
        and record["changed_direction_count"] == len(changed)
        and record["changed_participant_count"] == len({row["participant_id"] for row in changed})
        and record["outer_refit_needed"] is bool(changed), "complete frozen change set differs")
    return {row["participant_id"]: row for row in participants}


def _check_direction(row, original, selection):
    changed = selection["selection_changed"]
    require(row.get("selection_changed") is changed, "frozen change status differs")
    if not changed:
        require(row == {**original, "source": "reused_original", "selection_changed": False},
            "unchanged direction must reuse the original complete outcome exactly")
        return
    for key in ("status", "binding", "direction", "training_sequence_ids", "heldout_sequence_ids",
                "inner_validations_completed", "inner_mean_normalized_identity_losses"):
        require(row.get(key) == original[key], "original direction geometry or inner provenance differs")
    require(row.get("source") == "refitted_frozen_alternative"
        and row.get("selected_penalty_index") == selection["alternative_selected_penalty_index"]
        and row.get("selected_ridge_penalty") == selection["alternative_selected_ridge_penalty"]
        and row.get("alternative_pooled_normalized_identity_losses") == selection["alternative_pooled_normalized_identity_losses"]
        and row.get("selector_objective") == "pooled_inner_loss_ratio"
        and row.get("nuisance_losses_reproduced") is True
        and row.get("training_whitener_reproduced") is True
        and row.get("inner_retuning_performed") is False,
        "only the frozen alternative penalty may change this outer fit")
    raw = selector._numeric(row["identity_sequence_losses"], (6,))
    nuisance = selector._numeric(row["nuisance_sequence_losses"], (6,))
    require(nuisance.mean() > 0, "positive original outer nuisance denominator required")
    _match(nuisance, original["nuisance_sequence_losses"], "refitted nuisance losses differ from original")
    _match(1 - row["gain"], raw.mean() / nuisance.mean(), "alternative direction gain differs from losses")
    whitening, previous = row["training_whitening"], original["training_whitening"]
    require(whitening["sequence_ids"] == previous["sequence_ids"]
        and whitening["shrinkage_strength"] == previous["shrinkage_strength"], "training-only whitening support differs")
    _match(whitening["regularized_covariance_trace"], previous["regularized_covariance_trace"],
        "refitted training whitening differs from original")


def validate_completed_person(row, *, selection_record, original_control):
    require(row.get("schema") == SCHEMA and row.get("status") == STATUS
        and row.get("participant_id") == selection_record["participant_id"]
        and row.get("original_binding") == original_control["binding"] == selection_record["binding"]
        and row.get("frozen_selection_record") == selection_record
        and row.get("actual_reader_binding_validated") is True
        and row.get("inference_unit") == "participant"
        and all(row.get(key) is False for key in ("original_R_changed", "original_control_threshold_changed",
            "inner_retuning_performed", "primary_rescue", "audit_access", "concept_or_false_scores_computed")),
        "complete matching exploratory person with unchanged boundaries required")
    require([item.get("direction") for item in row["directions"]] == list(selector.DIRECTIONS),
        "both original directions required for complete person")
    for item, original, selection in zip(row["directions"], original_control["directions"], selection_record["directions"]):
        _check_direction(item, original, selection)
    _match(1 - row["gain"], 1 - np.mean([item["gain"] for item in row["directions"]]),
        "person gain must average both directions")
    require(row.get("original_gain") == original_control["gain"], "original person gain must remain unchanged")
    return row


def run_participant(person, *, selection_record, original_control, recipe, run_dir, budget_check, progress):
    import repeated_image_control as control
    from repeated_signal_diagnostic import _validate_completed
    from core_reference_direct_ridge import DirectCoreRidge
    from training_transforms import CommonSensorSpace

    recipe = control.validate_recipe(recipe)
    sequences = control._roster(person, budget_check)
    originals = _validate_completed(person, sequences, recipe, original_control)
    require(selection_record["participant_id"] == person.participant_id
        and selection_record["binding"] == original_control["binding"]
        and any(row["selection_changed"] for row in selection_record["directions"]),
        "original reader binding and at least one frozen changed direction required")
    for original, selection in zip(originals, selection_record["directions"]):
        require(original["selected_penalty_index"] == selection["original_selected_penalty_index"]
            and original["selected_ridge_penalty"] == selection["original_selected_ridge_penalty"]
            and original["inner_mean_normalized_identity_losses"] == selection["original_mean_normalized_identity_losses"],
            "original outer selection must match the independently frozen inner reconstruction")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sensor = CommonSensorSpace(sequences[0].channel_order)
    designs = {}
    for item in sequences:
        budget_check()
        designs[item.sequence_id] = control._sequence_designs(item)
    backend = DirectCoreRidge(check=budget_check)
    results = []
    try:
        for parity, (original, selection) in enumerate(zip(originals, selection_record["directions"])):
            budget_check()
            if not selection["selection_changed"]:
                results.append({**original, "source": "reused_original", "selection_changed": False})
                continue
            path = directory / (original["direction"] + ".complete.json")
            if path.exists():
                saved = control._json_read(path)
                _check_direction(saved, original, selection)
                results.append(saved)
                progress(original["direction"] + ":reused_complete_alternative")
                continue
            training = tuple(item for item in sequences if item.ordinal % 2 == parity)
            heldout = tuple(item for item in sequences if item.ordinal % 2 != parity)
            progress(original["direction"] + ":reproduce_original_nuisance")
            state = control._prepare_training(training, designs, sensor, recipe, backend, budget_check)
            nuisance = control._score(heldout, designs, sensor, state, "nuisance",
                state.nuisance_fit.coefficients, backend, budget_check)
            trace = float(np.trace(state.whitener.regularized_covariance))
            _match(nuisance, original["nuisance_sequence_losses"], "original nuisance losses must reproduce before alternative fit")
            _match(trace, original["training_whitening"]["regularized_covariance_trace"],
                "original training whitening must reproduce before alternative fit")
            progress(original["direction"] + ":fit_frozen_alternative_penalty")
            fit = control._identity_fit(state, selection["alternative_selected_penalty_index"], recipe, backend, budget_check)
            raw = control._score(heldout, designs, sensor, state, "identity", fit.coefficients, backend, budget_check)
            result = deepcopy(original)
            result.update(source="refitted_frozen_alternative", selection_changed=True,
                selected_penalty_index=selection["alternative_selected_penalty_index"],
                selected_ridge_penalty=selection["alternative_selected_ridge_penalty"],
                alternative_pooled_normalized_identity_losses=selection["alternative_pooled_normalized_identity_losses"],
                selector_objective="pooled_inner_loss_ratio", identity_sequence_losses=raw.tolist(),
                nuisance_sequence_losses=nuisance.tolist(), gain=float(1 - control._normal(raw, nuisance)),
                nuisance_losses_reproduced=True, training_whitener_reproduced=True,
                inner_retuning_performed=False,
                training_whitening={"sequence_ids": list(state.sequence_ids), "shrinkage_strength": state.whitener.shrinkage_strength,
                    "regularized_covariance_trace": trace}, solver=dict(backend.stats))
            _check_direction(result, original, selection)
            budget_check()
            control._json_write(path, result)
            results.append(result)
            del state, fit, raw, nuisance
            backend.clear()
            progress(original["direction"] + ":alternative_complete")
    finally:
        backend.clear()
    result = {"schema": SCHEMA, "status": STATUS, "participant_id": person.participant_id,
        "data_role": "development_repeated_signal_recovery_only", "original_binding": original_control["binding"],
        "frozen_selection_record": selection_record, "directions": results,
        "gain": float(np.mean([row["gain"] for row in results])), "original_gain": original_control["gain"],
        "actual_reader_binding_validated": True, "inference_unit": "participant",
        "cohort_inference_performed": False, "original_R_changed": False,
        "original_control_threshold_changed": False, "inner_retuning_performed": False,
        "primary_rescue": False, "audit_access": False, "concept_or_false_scores_computed": False}
    validate_completed_person(result, selection_record=selection_record, original_control=original_control)
    budget_check()
    control._json_write(directory / "complete.json", result)
    return result
