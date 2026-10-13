"""Fixed all-band residual ridge and added-block training-label controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_history import ridge_predictions
from primary_lmp_high import _contrast, prepare_split as prepare_J
from primary_pca import projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

BLOCKS = {"lmp": ((0., 0.),), "high": ((100., 200.), (200., 400.)),
          "added": ((.5, 4.), (4., 8.), (8., 12.), (12., 25.), (25., 50.), (50., 100.))}
ALL_BANDS = tuple(b for bands in BLOCKS.values() for b in bands)
RECIPES = ("lmp_all_band_joint_ridge_v1", "lmp_all_band_added_training_labels8_v1")
ANCHORS = {"joint": "lmp_high_joint_ridge_v1", "lmp": "lmp_full_channel_ridge_v1"}
REPLICATES = tuple(range(8))


def prepare_split(lmp, high, added, train):
    zl, zh = prepare_J(lmp, high, train)
    states = {name: {"physical_mean": block[train].mean(axis=(0, 1)),
                     "physical_sd": block[train].std(axis=(0, 1))}
              for name, block in (("lmp", lmp), ("high", high), ("added", added))}
    state = states["added"]
    za = (added-state["physical_mean"])/np.where(state["physical_sd"] == 0, 1., state["physical_sd"])
    return {"lmp": zl, "high": zh, "added": za}, states


def joint_arm(standardized, train, test, donors=None):
    added = standardized["added"][train]
    if donors is not None:
        if (np.asarray(donors).shape != (int(np.count_nonzero(train)),)
                or not np.array_equal(np.sort(donors), np.arange(len(donors)))):
            raise ValueError("Complete fitting-only added-block donor permutation required")
        added = added[donors]
    fitting = np.concatenate((standardized["lmp"][train], standardized["high"][train], added), axis=-1)
    evaluation = np.concatenate(tuple(standardized[name][test] for name in BLOCKS), axis=-1)
    mean = fitting.mean(axis=(0, 1))
    return fitting-mean, evaluation-mean, mean


def trial_map(ids, labels, session, held, replicate):
    ids, labels = np.asarray(ids), np.asarray(labels)
    numeric = numeric_trial_ids(ids, session)
    allowed = sorted(set(range(8))-{held})
    if (held not in range(8) or replicate not in REPLICATES or labels.shape != ids.shape
            or set(labels.tolist()) != set(allowed)):
        raise ValueError("Original IDs and exactly seven fitting directions required")
    mapping, records = np.empty(len(ids), dtype=int), []
    for direction in allowed:
        positions = sorted(np.flatnonzero(labels == direction).tolist(), key=lambda i: int(numeric[i]))
        donors = list(positions)
        literal = f"ep05-lmp-all-band-added-training-label-v1|20261004|{session}|{held}|{replicate}|{direction}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donors)
        mapping[positions] = donors
        records.append({"direction": direction, "seed_literal": literal,
            "recipient_trial_ids": ids[positions].tolist(), "donor_trial_ids": ids[donors].tolist()})
    return mapping, records


def average_control_banks(banks):
    if len(banks) != 8:
        raise ValueError("All eight fixed added-block controls required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError("Control target/baseline support differs: "+key)
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("All original penalties required")
        correct, wrong = (float(np.mean([row[key] for row in seeds])) for key in ("candidate_SSE", "wrong_pair_SSE"))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
        "added_training_replicate_ids": list(REPLICATES),
        "added_training_seed_rows": [{"replicate": r, "lambda_rows": bank["lambda_rows"]} for r, bank in zip(REPLICATES, banks)],
        "control_aggregation": "mean correct and exact wrong-pair SSE; not averaged predictions"}


def fit_banks(lmp, high, added, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    lmp, high, added, y = (np.asarray(value, dtype=float) for value in (lmp, high, added, targets))
    labels, ids, electrodes = np.asarray(directions), np.asarray(trial_ids), np.asarray(electrode_ids)
    numeric_trial_ids(ids, session)
    if (lmp.ndim != 3 or lmp.shape[:2] != (len(y), 15) or lmp.shape[-1] < 1
            or high.shape != (*lmp.shape[:2], 2*lmp.shape[-1]) or added.shape != (*lmp.shape[:2], 6*lmp.shape[-1])
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not all(np.isfinite(value).all() for value in (lmp, high, added, y))
            or set(labels.tolist()) != set(range(8)) or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or electrodes.shape != (lmp.shape[-1],) or not np.issubdtype(electrodes.dtype, np.integer)
            or np.any(electrodes < 1) or not np.array_equal(electrodes, np.unique(electrodes))):
        raise ValueError("Complete original all-band/native XY/physical/trial support required")
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    shuffled = np.full((8, len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        train, test = labels != held, labels == held
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        true_u = residual-intercept
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        standardized, states = prepare_split(lmp, high, added, train)
        fitting, evaluation, mean = joint_arm(standardized, train, test)
        paired_fit, paired_eval = fitting, evaluation
        added_energy = float(np.square(fitting[:, :, 3*lmp.shape[-1]:]).sum()/len(true_u))
        paired_covariance = projected_covariance(fitting)
        fitted = ridge_predictions(fitting, evaluation, true_u, check)
        for index in range(len(LAMBDAS)):
            paired[index, test] = base+fitted[index].reshape(base.shape)
        control_means, control_covariances, maps, equals = [], [], [], []
        for replicate in REPLICATES:
            check()
            donors, details = trial_map(ids[train], labels[train], session, held, replicate)
            fitting, evaluation, control_mean = joint_arm(standardized, train, test, donors)
            equals.append(bool(np.array_equal(fitting, paired_fit) and np.array_equal(evaluation, paired_eval)))
            control_means.append(control_mean.tolist())
            control_covariances.append(projected_covariance(fitting))
            maps.append({"replicate": replicate, "directions": details})
            fitted = ridge_predictions(fitting, evaluation, true_u, check)
            for index in range(len(LAMBDAS)):
                shuffled[replicate, index, test] = base+fitted[index].reshape(base.shape)
        metadata.append({"held_direction": held, "fitting_bin_rows": len(true_u),
            "physical_states": {name: {key: value.tolist() for key, value in state.items()} for name, state in states.items()},
            "paired_fitting_column_mean": mean.tolist(), "control_fitting_column_means": control_means,
            "added_fitting_energy": added_energy, "structural_J_duplicate": added_energy == 0,
            "paired_covariance": paired_covariance, "control_covariances": control_covariances,
            "training_added_maps": maps, "controls_equal_candidate_by_replicate": equals})
    duplicate = all(row["structural_J_duplicate"] for row in metadata)
    same = all(all(row["controls_equal_candidate_by_replicate"]) for row in metadata)
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.update(feature_width=9*lmp.shape[-1], physical_electrode_ids=electrodes.tolist(),
        bands_hz={name: [list(b) for b in bands] for name, bands in BLOCKS.items()},
        direction_fit_metadata=metadata, original_trial_ids=ids.tolist(), original_trial_directions=labels.tolist(),
        native_XY_scaled=False, block_rescaling=False, structural_J_duplicate=duplicate, all_controls_equal_candidate=same)
    control = average_control_banks([score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES])
    control.update(feature_width=9*lmp.shape[-1], structural_J_duplicate=duplicate, structural_paired_duplicate=same,
        training_added_only_permuted=True, evaluation_unpermuted=True, true_fitting_targets=True,
        physical_normalization_preserved=True, controls_have_own_full_Gram=True, diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panels):
    if set(sessions) != set(SESSIONS) or set(anchor_banks) != set(ANCHORS) or set(anchor_panels) != set(ANCHORS):
        raise ValueError("All original sessions and both preserved anchor families required")
    for name, recipe in ANCHORS.items():
        panel = anchor_panels[name]
        if (set(anchor_banks[name]) != set(SESSIONS) or panel["recipe"] != recipe
                or len(panel["sessions"]) != 16 or {r["session_id"] for r in panel["sessions"]} != set(SESSIONS)):
            raise ValueError("Complete J/LMP anchors required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete all-band banks required")
        for recipe, bank in [*banks.items(), *((recipe, anchor_banks[name][session]) for name, recipe in ANCHORS.items())]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"] for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original-penalty scores required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("All-band/control/anchor target support differs")
        if (banks[RECIPES[1]]["added_training_replicate_ids"] != list(REPLICATES)
                or banks[RECIPES[1]]["structural_J_duplicate"] != banks[RECIPES[0]]["structural_J_duplicate"]
                or banks[RECIPES[1]]["structural_paired_duplicate"] != banks[RECIPES[0]]["all_controls_equal_candidate"]):
            raise ValueError("All controls and consistent exact duplicate metadata required")
    paired = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, matched = {r: [] for r in RECIPES}, {name: [] for name in ANCHORS}
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(paired, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seeds = bank["added_training_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered control seeds required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            elif "direction_fit_metadata" in bank:
                def covariance(c):
                    return {"trace": c["trace"], "numerical_rank": c["numerical_rank"],
                            "effective_df": c["ridge_df_by_lambda"][chosen]["effective_df"]}
                row["selected_fit_diagnostics"] = [{"held_direction": d["held_direction"],
                    "added_fitting_energy": d["added_fitting_energy"], "structural_J_duplicate": d["structural_J_duplicate"],
                    "paired": covariance(d["paired_covariance"]), "controls": [covariance(c) for c in d["control_covariances"]],
                    "controls_equal_candidate_by_replicate": d["controls_equal_candidate_by_replicate"]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        for name in ANCHORS:
            matched[name].append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[name][held]["lambda_rows"][chosen]})
    duplicate = all(paired[s]["structural_J_duplicate"] for s in SESSIONS)
    same = all(paired[s]["all_controls_equal_candidate"] for s in SESSIONS)
    charges = {RECIPES[0]: int(not duplicate), RECIPES[1]: int(not duplicate and not same)}
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      scientific_trial_count_increment=charges[recipe], promotion_authorized=False,
                      search_complete=False, evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "All-band minus mean added-block training-label control")
    contrasts = {}
    for name in ANCHORS:
        contrasts[name+"_independently_selected"] = _contrast(selected[RECIPES[0]], anchor_panels[name]["sessions"], "All-band minus independently selected "+name)
        contrasts[name+"_all_band_lambda_matched"] = _contrast(selected[RECIPES[0]], matched[name], "All-band minus "+name+" at all-band-selected lambda")
    screens = {name: c["all_directional_screens_pass"] for name, c in contrasts.items()}
    screens["added_training_alignment"] = control["all_directional_screens_pass"]
    return {"panels": panels, "training_added_label_control": control, "anchor_contrasts": contrasts,
        "matched_lambda_anchor_sessions": matched, "complementarity_screens": screens,
        "all_complementarity_screens_pass": all(screens.values()), "structural_J_duplicate": duplicate,
        "all_controls_equal_candidate": same, "scientific_trial_count_by_recipe": charges,
        "scientific_trial_count_increment": sum(charges.values()), "control_trial_count_increment": charges[RECIPES[1]]}
