"""Other-animal donor restriction for the frozen learned electrode-value rule."""
import random

import numpy as np

from acquired_only_reader import balanced_ids
from learned_electrode_value import FEATURES, MC_SEEDS, VIEWS, mean_control_cells, rank_electrodes
from prepare_development_trial_roles import DEVELOPMENT

POLICY = "learned_electrode_value_reverse_animal_v1"
CONTROL = "learned_electrode_value_reverse_animal_labels3_v1"
ANIMALS = ("Chewie", "Mihili")
BRANCHES = ((POLICY, None), *((CONTROL, seed) for seed in MC_SEEDS))


def animal_support(session_animals):
    if (set(session_animals) != set(DEVELOPMENT)
            or any(sum(a == animal for a in session_animals.values()) != 4 for animal in ANIMALS)):
        raise ValueError("Exactly four fixed development sessions for each original animal required")


def training_records(packet, sources):
    """Whitelist saved training data, never outcome panels, as fitting inputs.

    The caller physically parses the combined completed report before fitting.
    This is a numerical input boundary, not an unopened-file or OS guarantee.
    """
    if (packet["status"] != "completed_two_learned_selector_comparisons"
            or packet["policy"] != "learned_electrode_value_linear_v1"
            or packet["control"] != "learned_electrode_value_labels3_v1"
            or packet["MC_seeds"] != list(MC_SEEDS) or packet["held_session_accessed"]
            or packet["combined_fit_progress"]["donor_label_fits"] != 7200):
        raise ValueError("Complete original learned-selector training record packet required")
    animals = {name: sources[name][1]["animal"] for name in DEVELOPMENT}
    animal_support(animals)
    rows = packet["donor_training_records"]
    expected = {(name, view) for name in DEVELOPMENT for view in VIEWS}
    if len(rows) != 40 or {(r["filename"], r["view"]) for r in rows} != expected:
        raise ValueError("Exactly all forty original donor information views required")
    fields = ("filename", "animal", "view", "channel_ids", "predictors", "value_labels",
              "fitting_rows", "later_label_rows", "label_denominator", "label_baseline_SSE",
              "target_unit_count", "donor_alpha", "donor_objective", "helper",
              "fitting_trial_ids", "later_label_trial_ids")
    result = []
    for row in rows:
        source, role, _ = sources[row["filename"]]
        view = row["view"]
        fitting = (balanced_ids(role, 32) if view == "outer" else
                   role["nested_training_sets"]["32"]["inner_folds"][int(view)]["fitting_trials"])
        later = [trial for pool in role["fitting_pool_by_direction"] for trial in pool[4:12]]
        if (row["animal"] != animals[row["filename"]]
                or row["channel_ids"] != source["common_physical_channels"]
                or len(row["channel_ids"]) != 96 or row["fitting_trial_ids"] != fitting
                or row["later_label_trial_ids"] != later or row["later_label_rows"] != 64
                or row["fitting_rows"] != (32 if view == "outer" else 24)
                or row["donor_alpha"] != 1.
                or row["donor_objective"] != "squared_Frobenius_residual/n + alpha*squared_Frobenius_coefficients"
                or row["helper"] != "ridge_training.ridge_coefficients; direction_training.residual_statistics/residual_coordinates"):
            raise ValueError("Saved donor labels differ from the adopted source roles/objective")
        result.append({key: row[key] for key in fields})
    inherited = {
        "full_record_pass_donor_fits": 3840, "historically_incurred_donor_fits": 7200,
        "new_donor_label_fits": 0, "donor_views_reused": 40,
        "donor_preparation_seconds_full_pass": packet["donor_preparation_seconds"],
        "full_pass_donor_source_cost": packet["donor_phase_cost"],
        "historical_combined_component_costs": packet["combined_component_costs"],
        "historical_combined_component_elapsed_seconds": packet["combined_component_elapsed_seconds"],
        "historical_full_report_bytes": packet["retained_report_bytes"],
        "per_target_donor_sessions": 4, "per_target_donor_fitting_trials": 384,
        "per_target_donor_LFP_electrode_trials": 4*96*96,
        "historical_cost_includes_prior_target_fits": True,
        "new_fit_cost_does_not_include_inherited_cost_again": True,
    }
    return result, animals, inherited


def fit_animal_ranker(records, recipient_animal, view, session_animals, replicate=None):
    animal_support(session_animals)
    view = str(view)
    if recipient_animal not in ANIMALS or view not in VIEWS or (replicate is not None and replicate not in MC_SEEDS):
        raise ValueError("Declared recipient animal, information view and optional control seed required")
    expected = {name for name, animal in session_animals.items() if animal != recipient_animal}
    # Metadata filters precede every numerical access: same-animal and wrong-
    # view predictors/labels cannot enter donor scales, coefficients or controls.
    selected = [r for r in records if r["filename"] in expected and str(r["view"]) == view]
    if len(selected) != 4 or {r["filename"] for r in selected} != expected:
        raise ValueError("Exactly four OTHER-animal donor sessions required")
    selected.sort(key=lambda r: r["filename"])
    xx, yy, weight_rows, permutations = [], [], [], []
    for row in selected:
        channels = row["channel_ids"]
        features, labels = np.asarray(row["predictors"], dtype=float), np.asarray(row["value_labels"], dtype=float)
        if (row["animal"] != session_animals[row["filename"]] or row["animal"] == recipient_animal
                or channels != sorted(set(channels)) or len(channels) != 96
                or features.shape != (96, 3) or labels.shape != (96,)
                or not np.isfinite(features).all() or not np.isfinite(labels).all()
                or row["fitting_rows"] != (32 if view == "outer" else 24)
                or not np.isfinite(row["label_denominator"]) or row["label_denominator"] <= 0
                or not np.isfinite(row["label_baseline_SSE"]) or row["label_baseline_SSE"] < 0
                or row["target_unit_count"] <= 0):
            raise ValueError("Finite complete ninety-six-electrode permitted donor view required")
        if replicate is not None:
            literal = f"ep08-learned-electrode-label-v1|20261003|{row['filename']}|{view}|{replicate}"
            order = list(range(96))
            rng = random.Random()
            rng.seed(literal, version=2)
            rng.shuffle(order)
            labels = labels[order]
            permutations.append({"donor_filename": row["filename"], "view": view, "replicate": replicate,
                "seed_literal": literal, "recipient_channel_ids": channels,
                "donor_channel_ids": [channels[index] for index in order]})
        xx.append(features)
        yy.append(labels)
        weight_rows.append({"filename": row["filename"], "animal": row["animal"],
                            "electrode_weight": 1./384, "session_weight": .25})
    xx, yy = np.vstack(xx), np.concatenate(yy)
    weights = np.full(384, 1./384)
    weights /= weights.sum()
    mean = np.sum(weights[:, None]*xx, axis=0)
    raw_sd = np.sqrt(np.sum(weights[:, None]*(xx-mean)**2, axis=0))
    sd = np.where(raw_sd == 0, 1., raw_sd)
    z = (xx-mean)/sd
    zm, ym = np.sum(weights[:, None]*z, axis=0), float(weights@yy)
    centered = z-zm
    coefficient = np.linalg.solve(centered.T@(weights[:, None]*centered)+np.eye(3),
                                   centered.T@(weights*(yy-ym)))
    intercept = ym-float(zm@coefficient)
    return {"recipient_animal": recipient_animal, "view": view, "replicate": replicate,
        "predictor_names": list(FEATURES), "predictor_mean": mean.tolist(), "predictor_sd": sd.tolist(),
        "zero_SD_predictors": np.flatnonzero(raw_sd == 0).tolist(), "coefficients": coefficient.tolist(),
        "intercept": intercept, "penalty": 1.,
        "objective": "weighted mean squared label error + squared coefficient norm; unpenalized intercept",
        "donor_sessions": [r["filename"] for r in selected], "record_weights": weights.tolist(),
        "donor_weights": weight_rows, "label_permutations": permutations}


def build_ranker_bank(records, session_animals, progress, check=lambda: None):
    bank = {}
    for animal in ANIMALS:
        for view in VIEWS:
            for replicate in (None, *MC_SEEDS):
                check()
                bank[animal, view, replicate] = fit_animal_ranker(records, animal, view, session_animals, replicate)
                progress["reverse_ranker_fits"] += 1
    return bank
