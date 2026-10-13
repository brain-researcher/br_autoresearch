"""Fixed LMP history ridge and matched per-trial channel-label control."""
from dataclasses import dataclass
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_history4_ridge_v1", "lmp_history4_channel_labels8_v1")
LAGS = (0, -1, -2, -3)
REPLICATES = tuple(range(8))


def history_blocks(x):
    """Input offsets -3:+14; output axes trial,target time,lag,electrode."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or x.shape[1] != 18 or x.shape[2] < 1:
        raise ValueError("Exact eighteen-bin complete LMP source history required")
    return np.stack([x[:, 3+lag:18+lag, :] for lag in LAGS], axis=2)


@dataclass
class HistoryState:
    feature_mean: np.ndarray
    feature_sd: np.ndarray


def standardize_history(blocks, state):
    return (blocks-state.feature_mean)/np.where(state.feature_sd == 0, 1., state.feature_sd)


def channel_maps(ids, electrode_count, session_id, replicate):
    ids = np.asarray(ids)
    if (replicate not in REPLICATES or electrode_count < 1 or ids.ndim != 1
            or len(set(map(str, ids.tolist()))) != len(ids)):
        raise ValueError("Unique original trial tokens, physical electrodes and fixed replicate required")
    rows = []
    for trial in ids:
        mapping = list(range(electrode_count))
        rng = random.Random()
        rng.seed(f"ep05-lmp-history-channel-label-v1|20261003|{session_id}|{trial}|{replicate}", version=2)
        rng.shuffle(mapping)
        rows.append(mapping)
    return np.asarray(rows, dtype=int).reshape(len(ids), electrode_count)


def permute_channels(blocks, maps):
    if (blocks.ndim != 4 or blocks.shape[1:3] != (15, 4)
            or maps.shape != (len(blocks), blocks.shape[-1])
            or not np.issubdtype(maps.dtype, np.integer)
            or not np.all(np.sort(maps, axis=1) == np.arange(blocks.shape[-1]))):
        raise ValueError("Each trial needs one complete electrode permutation shared across all times/lags")
    return np.take_along_axis(blocks, maps[:, None, None, :], axis=3)


def center_arm(fitting, evaluation):
    train = fitting.reshape(len(fitting), 15, -1)
    test = evaluation.reshape(len(evaluation), 15, -1)
    mean = train.mean(axis=(0, 1))
    return train-mean, test-mean, mean


def ridge_predictions(fitting, evaluation, residual, check=lambda: None):
    """A separate Gram matrix for each channel-label arm; five fixed penalties."""
    phi = fitting.reshape(-1, fitting.shape[-1])
    test = evaluation.reshape(-1, evaluation.shape[-1])
    gram = phi.T@phi
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    rhs = vectors.T@(phi.T@residual)
    predictions = np.empty((len(LAMBDAS), len(test), 2))
    for index, penalty in enumerate(LAMBDAS):
        check()
        denominator = values+len(phi)*penalty
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ArithmeticError("Invalid positive-penalty history eigensystem")
        predictions[index] = test@(vectors@(rhs/denominator[:, None]))
    return predictions


def prepare_split(blocks, y, labels, held):
    train, test = labels != held, labels == held
    template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
    residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
    intercept = residual.mean(axis=0)
    state = HistoryState(blocks[train].mean(axis=(0, 1)), blocks[train].std(axis=(0, 1)))
    return {"train": train, "test": test, "template": template, "intercept": intercept,
            "residual": residual-intercept, "history_state": state,
            "fitting_standardized": standardize_history(blocks[train], state)}


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed physical-channel-label replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Label control replicate support differs: {key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("Label control penalty identities differ")
        correct = float(np.mean([row["candidate_SSE"] for row in seeds]))
        wrong = float(np.mean([row["wrong_pair_SSE"] for row in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
                     "candidate_SSE": correct, "wrong_pair_SSE": wrong,
                     "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
                     "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
            "channel_label_replicate_ids": list(REPLICATES), "channel_label_Monte_Carlo_draws": 8,
            "channel_label_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]}
                                        for r, b in zip(REPLICATES, banks)],
            "channel_label_aggregation": "mean correct/wrong SSE; not averaged predictions",
            "lambda_selection": "same paired outer-training-selected lambda for all eight label refits"}


def fit_banks(features, targets, directions, trial_ids, session_id, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[2] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or len(set(map(str, ids.tolist()))) != len(y) or set(labels.tolist()) != set(range(8))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == direction) < 2 for direction in range(8))):
        raise ValueError("Complete original history/target/electrode/trial support required")
    blocks = history_blocks(x)
    maps = np.stack([channel_maps(ids, x.shape[-1], session_id, r) for r in REPLICATES])
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    randomized = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_split(blocks, y, labels, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        train_blocks = split["fitting_standardized"]
        test_blocks = standardize_history(blocks[test], state)
        fitting, evaluation, paired_mean = center_arm(train_blocks, test_blocks)
        fitted = ridge_predictions(fitting, evaluation, split["residual"], check)
        for index in range(len(LAMBDAS)):
            paired[index, test] = base+fitted[index].reshape(base.shape)
        control_means = []
        for replicate in REPLICATES:
            check()
            fitting, evaluation, mean = center_arm(permute_channels(train_blocks, maps[replicate, train]),
                                                   permute_channels(test_blocks, maps[replicate, test]))
            fitted = ridge_predictions(fitting, evaluation, split["residual"], check)
            for index in range(len(LAMBDAS)):
                randomized[replicate, index, test] = base+fitted[index].reshape(base.shape)
            control_means.append(mean.tolist())
        metadata.append({"held_direction": held, "fitting_bin_rows": len(split["residual"]),
                         "physical_lag_training_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
                         "paired_fitting_column_mean": paired_mean.tolist(),
                         "control_fitting_column_means_by_replicate": control_means})
    paired_bank = score_bank(y, labels, baseline, paired, RECIPES[0])
    paired_bank.update(lag_offsets=list(LAGS), direction_fit_metadata=metadata,
                       column_order="lag0,-1,-2,-3 then frozen physical electrode",
                       physical_normalization="lag/electrode mean and populationSD from fitting directions; zeroSD->1",
                       channel_label_trial_maps={"trial_ids": ids.tolist(), "replicate_ids": list(REPLICATES),
                           "source_channel_positions_by_replicate": maps.tolist(),
                           "position_basis": "zero-based positions in source.common_LMP_electrodes; destination-to-source"})
    controls = [score_bank(y, labels, baseline, randomized[r], RECIPES[1]) for r in REPLICATES]
    control_bank = average_control_banks(controls)
    control_bank.update(lag_offsets=list(LAGS), channel_label_map_metadata="paired bank channel_label_trial_maps",
                        physical_normalization_preserved=True, global_Gram_matrix_preserved=False,
                        evaluation_targets="original XY; channel maps apply only to features")
    return {RECIPES[0]: paired_bank, RECIPES[1]: control_bank}, baseline, {RECIPES[0]: paired, RECIPES[1]: randomized}


def summarize_banks(sessions):
    if set(sessions) != set(SESSIONS):
        raise ValueError("All sixteen sessions required for fixed history/channel-label comparison")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError(f"Both complete history banks required: {session}")
        for recipe in RECIPES:
            bank = banks[recipe]
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS) or bank["lag_offsets"] != list(LAGS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(row[key]) for row in bank["lambda_rows"]
                               for key in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError(f"Invalid complete history score bank: {session}/{recipe}")
        if banks[RECIPES[1]]["channel_label_replicate_ids"] != list(REPLICATES):
            raise ValueError("Every fixed channel-label replicate required")
        for key in ("baseline_SSE", "baseline_R2", "denominator", "direction_counts"):
            if banks[RECIPES[0]][key] != banks[RECIPES[1]][key]:
                raise ValueError("History candidate/control baseline or scoring support differs")
    paired_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected = {r: [] for r in RECIPES}
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(paired_banks, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                   "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                   "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seed_rows = bank["channel_label_seed_rows"]
                if [r["replicate"] for r in seed_rows] != list(REPLICATES):
                    raise ValueError("Channel-label seed score support incomplete")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]}
                                               for r in seed_rows]
                row["seed_primary_range"] = [min(r["delta_R2"] for r in row["selected_seed_scores"]),
                                              max(r["delta_R2"] for r in row["selected_seed_scores"])]
            selected[recipe].append(row)
    panels = []
    for recipe in RECIPES:
        summary = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": summary["primary_increment"] >= .01,
                   "identity_at_least_0p005": summary["identity_increment"] >= .005,
                   **directional_screens(summary)}
        summary.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                       scientific_trial_count_increment=1, promotion_authorized=False, search_complete=False,
                       role="candidate" if recipe == RECIPES[0] else "matched physical-channel-label control",
                       evidence="exposed exploratory development")
        panels.append(summary)
    rows = [{"session_id": a["session_id"], "animal": a["animal"], "lambda": a["lambda"],
             "delta_R2": a["delta_R2"]-b["delta_R2"],
             "identity_increment": a["identity_increment"]-b["identity_increment"]}
            for a, b in zip(selected[RECIPES[0]], selected[RECIPES[1]])]
    control = aggregate_rows(rows)
    screens = directional_screens(control)
    control.update(contrast="history minus eight-seed mean channel-label control",
                   directional_label_screens=screens, all_directional_label_screens_pass=all(screens.values()),
                   scientific_trial_count_increment=0, promotion_authorized=False,
                   interpretation="stable electrode labels after physical normalization; control changes global covariance")
    return panels, control
