"""Freeze one aggregation sensitivity using only completed inner-fold losses.

Original mean-of-ratios and alternative ratio-of-sums use the same six inner
folds, thirteen penalties and larger-penalty tie rule. No outer losses, EEG,
concept/false scores or audit records are read. This is exploratory development
sensitivity after the failed original control, not a replacement primary test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

import numpy as np


EPISODE = Path(__file__).resolve().parents[2]
INPUT_ROOT = EPISODE / "outputs/runs/repeated_image_signal_recovery/control001"
RECIPE_PATH = EPISODE / "outputs/config/core_recipe_20261013_001.json"
PEOPLE = tuple("sub-" + suffix for suffix in (
    "07", "08", "09", "10", "11", "12", "13", "14", "20", "21", "22", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "34", "35", "38", "43", "44", "45", "46", "47", "48", "49"))
DIRECTIONS = ("odd_to_even", "even_to_odd")
RIDGE_GRID = tuple(10.0 ** power for power in range(-6, 7))
SCHEMA = "ep18_repeated_selector_sensitivity_v1"
ORIGINAL_IDENTITY_RECIPE = {
    "identity_encoding": "one_hot_uncentered_200",
    "temporal_basis": "raised_cosine_12",
    "outer_folds": "chronological_physical_sequences_odd_to_even_and_even_to_odd",
    "inner_folds": "leave_one_training_physical_sequence_out",
    "inner_selection": "mean_inner_ratio_of_equal_sequence_means_then_larger_lambda",
    "ridge_grid": list(RIDGE_GRID),
    "direction_gain": "1_minus_ratio_of_equal_sequence_mean_whitened_MSEs",
    "participant_gain": "equal_mean_of_both_direction_gains",
    "drift_span_seconds": "last_event_onset_plus_0.801_minus_E3_cue",
}
SELECTION_RECIPE = {
    "choice_count": 1,
    "original_objective": "mean_of_six_identity_to_nuisance_inner_loss_ratios",
    "alternative_objective": "sum_of_six_identity_inner_losses_over_sum_of_six_nuisance_inner_losses",
    "alternative_weights_on_original_ratios": "inner_nuisance_loss_over_sum_of_six_inner_nuisance_losses",
    "tie_break": "minimum_objective_then_larger_lambda_then_index",
    "ridge_grid": list(RIDGE_GRID),
    "whitening": "each_inner_fold_keeps_its_own_original_training_fitted_whitener",
    "selection_scope": "all_30_people_both_directions_before_any_new_outer_fit",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    require(path.stat().st_size <= 16 * 1024**2, "bounded completed inner record required")
    with path.open() as source:
        return json.load(source)


def _numeric(value, shape):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in "iuf"
        and np.isfinite(result).all() and np.all(result >= 0),
        "complete finite nonnegative inner loss array required")
    return result


def _binding(value, person, recipe):
    require(value.get("schema") == "ep18_repeated_image_control_v1"
        and value.get("participant_id") == person and value.get("data_role") == "development"
        and value.get("recipe") == recipe
        and value.get("identity_recipe") == ORIGINAL_IDENTITY_RECIPE,
        "unchanged original participant, role and numerical recipe required")
    for key, count in (("image_ids", 200), ("channel_order", 62)):
        values = value[key]
        require(isinstance(values, list) and len(values) == len(set(values)) == count
            and all(isinstance(item, str) and item for item in values),
            "complete original image and sensor axes required")
    support = value["sequence_support"]
    require(len(support) == 12 and [row["ordinal"] for row in support] == list(range(12))
        and len({row["sequence_id"] for row in support}) == 12
        and all(isinstance(row["sequence_id"], str) and row["sequence_id"]
            and type(row["ordinal"]) is int and type(row["samples"]) is int
            and type(row["retained_samples"]) is int
            and 0 < row["retained_samples"] <= row["samples"] for row in support),
        "all twelve original chronological sequence supports required")
    return support


def _choice(objective):
    require(objective.shape == (13,) and np.isfinite(objective).all()
        and np.all(objective >= 0), "all thirteen finite selector objectives required")
    return min(range(13), key=lambda index: (objective[index], -RIDGE_GRID[index], index))


def _participant(root, person, recipe):
    binding = None
    directions = []
    for parity, direction in enumerate(DIRECTIONS):
        raw_losses, nuisance_losses, ratios, fold_rows = [], [], [], []
        for ordinal in range(parity, 12, 2):
            filename = f"{direction}_inner_{ordinal:02d}.json"
            record = _read(root / person / filename)
            require(record.get("status") == "complete_inner", "complete inner checkpoint required")
            if binding is None:
                binding = record["binding"]
                support = _binding(binding, person, recipe)
            require(record.get("binding") == binding, "same original binding in all twelve inner checkpoints required")
            outer_training = [row for row in support if row["ordinal"] % 2 == parity]
            omitted = support[ordinal]["sequence_id"]
            train_ids = [row["sequence_id"] for row in outer_training if row["ordinal"] != ordinal]
            require(record.get("training_sequence_ids") == train_ids
                and record.get("heldout_sequence_ids") == [omitted],
                "original five training sequences and one held-out inner sequence required")
            raw = _numeric(record["identity_sequence_losses"], (13, 1))
            nuisance = _numeric(record["nuisance_sequence_losses"], (1,))
            require(float(np.mean(nuisance)) > 0, "strictly positive original inner nuisance denominator required")
            normalized = _numeric(record["normalized_identity_losses"], (13,))
            # Same operations as continuous_tuning.normalize_losses, without
            # importing the EEG/model stack or reading any outer checkpoint.
            expected = np.mean(raw, axis=-1) / float(np.mean(nuisance))
            require(np.array_equal(normalized, expected), "saved original inner normalization differs")
            raw_losses.append(raw[:, 0])
            nuisance_losses.append(float(nuisance[0]))
            ratios.append(normalized)
            fold_rows.append({"checkpoint": f"{person}/{filename}", "heldout_ordinal": ordinal,
                "training_sequence_ids": train_ids, "heldout_sequence_id": omitted})
        raw, nuisance = np.asarray(raw_losses), np.asarray(nuisance_losses)
        original = np.mean(ratios, axis=0)
        denominator = float(np.sum(nuisance))
        require(np.isfinite(denominator) and denominator > 0,
            "finite positive pooled nuisance denominator required")
        pooled = np.sum(raw, axis=0) / denominator
        old_index, new_index = _choice(original), _choice(pooled)
        directions.append({"direction": direction,
            "outer_training_sequence_ids": [row["sequence_id"] for row in outer_training],
            "inner_folds": fold_rows, "inner_validations_completed": 6,
            "identity_inner_losses_by_fold_and_penalty": raw.tolist(),
            "nuisance_inner_losses_by_fold": nuisance.tolist(),
            "alternative_weights_on_inner_ratios": (nuisance / denominator).tolist(),
            "original_mean_normalized_identity_losses": original.tolist(),
            "alternative_pooled_normalized_identity_losses": pooled.tolist(),
            "original_selected_penalty_index": old_index,
            "original_selected_ridge_penalty": RIDGE_GRID[old_index],
            "alternative_selected_penalty_index": new_index,
            "alternative_selected_ridge_penalty": RIDGE_GRID[new_index],
            "selection_changed": old_index != new_index})
    return {"participant_id": person, "binding": binding, "directions": directions}


def report(*, input_root, output_dir):
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(),
        "saved-loss selection arithmetic requires a coordinated compute allocation")
    input_root, output_dir = Path(input_root).resolve(), Path(output_dir).resolve()
    require(input_root == INPUT_ROOT.resolve(), "only the original completed repeated-control root is released")
    require(not output_dir.exists() and (EPISODE / "outputs/runs").resolve() in output_dir.parents
        and input_root not in output_dir.parents,
        "fresh episode output directory outside the frozen source root required")
    recipe = _read(RECIPE_PATH)
    participants = [_participant(input_root, person, recipe) for person in PEOPLE]
    vector = [{"participant_id": row["participant_id"],
        **{key: direction[key] for key in ("direction", "original_selected_penalty_index",
            "original_selected_ridge_penalty", "alternative_selected_penalty_index",
            "alternative_selected_ridge_penalty", "selection_changed")}}
        for row in participants for direction in row["directions"]]
    changed = [row for row in vector if row["selection_changed"]]
    result = {"schema": SCHEMA, "status": "complete_frozen_repeated_selector_sensitivity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "input_root": str(input_root),
        "data_role": "development_repeated_signal_recovery_only",
        "participants": list(PEOPLE), "participant_records": participants,
        "participant_count": 30, "direction_count": 60, "inner_checkpoint_count": 360,
        "ridge_penalty_count": 13, "selection_recipe": SELECTION_RECIPE,
        "selection_vector": vector, "selection_vector_complete": True,
        "changed_direction_count": len(changed), "changed_directions": changed,
        "changed_participant_count": len({row["participant_id"] for row in changed}),
        "outer_refit_needed": bool(changed),
        "next_action": ("refit_only_changed_directions_using_this_complete_frozen_vector"
                        if changed else "no_outer_refit_needed_all_sixty_choices_unchanged"),
        "original_choices_reconstructed_from_inner_records": True,
        "original_outer_selection_records_read": False, "outer_losses_read": False,
        "EEG_read": False, "outer_refits_performed": False,
        "original_R_changed": False, "original_control_threshold_changed": False,
        "primary_rescue": False, "audit_access": False,
        "concept_or_false_scores_computed": False, "scientific_inference_performed": False,
        "interpretation": "Exploratory selector sensitivity after original control outcomes; not an objective bug correction.",
        "whitening_caveat": "Pooling weights the six old ratios by their nuisance losses; each fold has its own training-fitted whitener, so the alternative does not recreate the single outer-fit metric or establish a uniquely correct risk."}
    summary = "\n".join((
        "EP18 exploratory aggregation sensitivity: all 360 saved inner checkpoints, 30 people, 60 directions and 13 penalties; no EEG or outer losses read.",
        f"Pooled inner losses change {len(changed)}/60 selections in {result['changed_participant_count']}/30 people; all original and alternative choices are frozen together before any new outer refit.",
        ("Next: refit only the changed directions using this frozen vector. " if changed else "No outer refit is needed because all selections are unchanged. ")
        + "Fold-specific whiteners remain distinct; original R/G/T, thresholds and closed data roles remain unchanged, with no primary rescue."
    )) + "\n"
    # Publish no complete selection vector until every original person/fold has
    # passed. An interrupted write leaves no misleading complete JSON file.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.md").write_text(summary)
    temporary = output_dir / "repeated_selector_sensitivity.complete.json.tmp"
    with temporary.open("x") as target:
        json.dump(result, target, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(output_dir / "repeated_selector_sensitivity.complete.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = report(input_root=args.input_root, output_dir=args.output_dir)
    print(json.dumps({key: result[key] for key in
        ("status", "changed_direction_count", "changed_participant_count", "outer_refit_needed")}))


if __name__ == "__main__":
    main()
