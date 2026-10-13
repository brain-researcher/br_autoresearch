"""Adopted EP05 time-varying LMP ridge and matched training-pairing control."""
from dataclasses import dataclass
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, animal_balanced

RECIPES = ("lmp_timepoly2_ridge_v1", "lmp_timepoly2_train_derangement8_v1")
REPLICATES = tuple(range(8))
DERANGEMENT_DRAW_BOUND = 10000


def time_basis():
    u = np.arange(15, dtype=float)-7
    square_centered = u*u-np.mean(u*u)
    return np.column_stack((np.ones(15), u/np.sqrt(np.mean(u*u)),
                            square_centered/np.sqrt(np.mean(square_centered**2))))


@dataclass
class TensorState:
    channel_mean: np.ndarray
    channel_sd: np.ndarray
    product_mean: np.ndarray


def raw_tensor(x, mean, sd):
    z = (np.asarray(x)-mean)/np.where(sd == 0, 1., sd)
    # Trial, time, degree, electrode; preserve degree-major electrode ordering.
    return (z[:, :, None, :]*time_basis()[None, :, :, None]).reshape(len(z), 15, -1)


def fit_tensor(x):
    flat = np.asarray(x).reshape(-1, x.shape[-1])
    mean, sd = flat.mean(axis=0), flat.std(axis=0)
    product = raw_tensor(x, mean, sd)
    state = TensorState(mean, sd, product.mean(axis=(0, 1)))
    return product-state.product_mean, state


def transform_tensor(x, state):
    return raw_tensor(x, state.channel_mean, state.channel_sd)-state.product_mean


def training_derangement(ids, labels, session_id, held, replicate):
    """For each target trial position return a different source trial position."""
    ids, labels = np.asarray(ids), np.asarray(labels)
    legal = set(range(8))-{held}
    if (replicate not in REPLICATES or len(set(ids.tolist())) != len(ids)
            or ids.shape != labels.shape or set(labels.tolist()) != legal):
        raise ValueError("Unique fitting IDs, seven legal directions and fixed replicate required")
    result = np.arange(len(ids))
    for direction in sorted(legal):
        ordered = sorted(np.flatnonzero(labels == direction), key=lambda i: str(ids[i]))
        if len(ordered) < 2:
            raise ValueError("Each training direction requires two complete trials")
        literal = (f"ep05-lmp-timepoly2-train-derangement-v1|20261003|"
                   f"{session_id}|{held}|{direction}|{replicate}")
        rng = random.Random()
        rng.seed(literal, version=2)
        for _ in range(DERANGEMENT_DRAW_BOUND):
            permuted = list(range(len(ordered)))
            rng.shuffle(permuted)
            if all(i != j for i, j in enumerate(permuted)):
                result[ordered] = np.asarray(ordered)[permuted]
                break
        else:
            raise RuntimeError("Fixed derangement draw bound exhausted; no reseed")
    return result


def prepare_split(x, y, labels, ids, session_id, held):
    train, test = labels != held, labels == held
    angles = labels*np.pi/4
    template = DirectionTime.fit(angles[train], y[train], 1, labels[train])
    residual = (y[train]-template.predict(angles[train])).reshape(-1, 2)
    intercept = residual.mean(axis=0)
    fitting, state = fit_tensor(x[train])
    permutations = [training_derangement(ids[train], labels[train], session_id, held, r)
                    for r in REPLICATES]
    return {"train": train, "test": test, "template": template, "intercept": intercept,
            "residual": residual-intercept, "features": fitting, "tensor_state": state,
            "permutations": permutations}


def shared_ridge_predictions(fitting, evaluation, residual, permutations, check=lambda: None):
    """One shared design covariance; separate actual/shuffled target cross-products."""
    phi = fitting.reshape(-1, fitting.shape[-1])
    test = evaluation.reshape(-1, evaluation.shape[-1])
    gram = phi.T@phi
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    outputs = np.empty((1+len(permutations), len(LAMBDAS), len(test), 2))
    for arm in range(1+len(permutations)):
        check()
        design = phi if arm == 0 else fitting[permutations[arm-1]].reshape(phi.shape)
        rhs = vectors.T@(design.T@residual)
        for index, penalty in enumerate(LAMBDAS):
            check()
            denominator = values+len(phi)*penalty
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ArithmeticError("Nonpositive/nonfinite penalized design eigensystem")
            coefficient = vectors@(rhs/denominator[:, None])
            outputs[arm, index] = test@coefficient
    return outputs


def average_control_banks(banks):
    """Mean seed SSEs; never score an ensemble of the seed predictions."""
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed training-shuffle replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Control replicate score support differs: {key}")
    rows = []
    for i, penalty in enumerate(LAMBDAS):
        seed_rows = [b["lambda_rows"][i] for b in banks]
        if any(row["lambda"] != penalty for row in seed_rows):
            raise ValueError("Control penalty identities differ")
        mean_sse = float(np.mean([r["candidate_SSE"] for r in seed_rows]))
        mean_wrong = float(np.mean([r["wrong_pair_SSE"] for r in seed_rows]))
        denominator = first["denominator"]
        rows.append({"lambda": penalty, "candidate_R2": 1-mean_sse/denominator,
                     "candidate_SSE": mean_sse, "wrong_pair_SSE": mean_wrong,
                     "delta_R2": (first["baseline_SSE"]-mean_sse)/denominator,
                     "identity_increment": (mean_wrong-mean_sse)/denominator})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
            "training_shuffle_replicate_ids": list(REPLICATES),
            "training_shuffle_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]}
                                          for r, b in zip(REPLICATES, banks)],
            "training_shuffle_aggregation": "mean SSE and mean wrong-pair SSE; not averaged predictions",
            "training_shuffle_Monte_Carlo_draws": 8,
            "lambda_selection": "same paired outer-training-selected lambda for all eight control refits"}


def fit_banks(features, targets, directions, trial_ids, session_id, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[2] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or len(set(ids.tolist())) != len(y) or set(labels.tolist()) != set(range(8))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original trial/window/electrode/direction support required")
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    shuffled = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_split(x, y, labels, ids, session_id, held)
        train, test, state = split["train"], split["test"], split["tensor_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        evaluation = transform_tensor(x[test], state)
        residual_predictions = shared_ridge_predictions(split["features"], evaluation,
                                                        split["residual"], split["permutations"], check)
        for index in range(len(LAMBDAS)):
            paired[index, test] = base+residual_predictions[0, index].reshape(base.shape)
            for replicate in REPLICATES:
                shuffled[replicate, index, test] = base+residual_predictions[replicate+1, index].reshape(base.shape)
        fitting_ids = ids[train]
        metadata.append({"held_direction": held, "fitting_bin_rows": len(split["residual"]),
            "training_zero_SD_columns": np.flatnonzero(state.channel_sd == 0).tolist(),
            "target_trial_ids_in_fitting_order": fitting_ids.tolist(),
            "target_directions_in_fitting_order": labels[train].tolist(),
            "training_feature_source_trial_ids_by_replicate": [
                {"replicate": r, "source_trial_ids": fitting_ids[p].tolist()}
                for r, p in zip(REPLICATES, split["permutations"])]})
    paired_bank = score_bank(y, labels, baseline, paired, RECIPES[0])
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    paired_bank.update(direction_fit_metadata=metadata, time_basis_values=time_basis().tolist(),
                       tensor_column_order="basis degree then original physical electrode",
                       tensor_centering="training-only product means; no product rescaling")
    control_bank = average_control_banks(controls)
    control_bank.update(trial_mapping_metadata="paired bank direction_fit_metadata",
                        evaluation_pairing="original correct trials; evaluation XY unchanged")
    return {RECIPES[0]: paired_bank, RECIPES[1]: control_bank}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def select_paired_lambda(paired_banks, held):
    training = [s for s in SESSIONS if s != held]
    scores = [animal_balanced({s: paired_banks[s]["lambda_rows"][i]["delta_R2"] for s in training})[0]
              for i in range(len(LAMBDAS))]
    chosen = max(range(len(LAMBDAS)), key=lambda i: (scores[i], LAMBDAS[i]))
    return chosen, training, scores


def aggregate_rows(rows):
    primary, primary_animals = animal_balanced({r["session_id"]: r["delta_R2"] for r in rows})
    identity, identity_animals = animal_balanced({r["session_id"]: r["identity_increment"] for r in rows})
    majority = {a: sum(r["delta_R2"] > 0 for r in rows if r["animal"] == a) for a in ("Mihili", "Chewie")}
    influence = [{"excluded_session": s,
        "primary_increment": animal_balanced({r["session_id"]: r["delta_R2"] for r in rows if r["session_id"] != s})[0],
        "identity_increment": animal_balanced({r["session_id"]: r["identity_increment"] for r in rows if r["session_id"] != s})[0]}
        for s in SESSIONS]
    return {"sessions": rows, "primary_increment": primary, "identity_increment": identity,
            "primary_by_animal": primary_animals, "identity_by_animal": identity_animals,
            "primary_positive_session_counts": majority,
            "leave_one_session_fixed_prediction_influence": influence}


def directional_screens(summary):
    majority = summary["primary_positive_session_counts"]
    return {"both_primary_animals_positive": all(v > 0 for v in summary["primary_by_animal"].values()),
            "both_identity_animals_positive": all(v > 0 for v in summary["identity_by_animal"].values()),
            "strict_session_majorities": majority["Mihili"] >= 4 and majority["Chewie"] >= 6}


def summarize_banks(sessions):
    if set(sessions) != set(SESSIONS):
        raise ValueError("All sixteen sessions required for paired temporal/control comparison")
    for s, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError(f"Both temporal/control banks required: {s}")
        for recipe in RECIPES:
            bank = banks[recipe]
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[key]) for r in bank["lambda_rows"]
                               for key in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError(f"Invalid temporal/control score bank: {s}/{recipe}")
        if banks[RECIPES[1]]["training_shuffle_replicate_ids"] != list(REPLICATES):
            raise ValueError("Complete fixed control replicate set required")
        for key in ("baseline_SSE", "baseline_R2", "denominator", "direction_counts"):
            if banks[RECIPES[0]][key] != banks[RECIPES[1]][key]:
                raise ValueError("Temporal/control opportunity or baseline differs")
    paired_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected_rows = {r: [] for r in RECIPES}
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(paired_banks, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                   "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                   "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seed_rows = bank["training_shuffle_seed_rows"]
                if [r["replicate"] for r in seed_rows] != list(REPLICATES):
                    raise ValueError("Control seed results incomplete or reordered")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]}
                                               for r in seed_rows]
                row["seed_primary_range"] = [min(r["delta_R2"] for r in row["selected_seed_scores"]),
                                              max(r["delta_R2"] for r in row["selected_seed_scores"])]
            selected_rows[recipe].append(row)
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected_rows[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005,
                   **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      scientific_trial_count_increment=1, promotion_authorized=False, search_complete=False,
                      role="candidate" if recipe == RECIPES[0] else "matched alternative-explanation control",
                      evidence="exposed exploratory development")
        panels.append(result)
    contrasts = [{"session_id": a["session_id"], "animal": a["animal"], "lambda": a["lambda"],
                  "delta_R2": a["delta_R2"]-b["delta_R2"],
                  "identity_increment": a["identity_increment"]-b["identity_increment"]}
                 for a, b in zip(selected_rows[RECIPES[0]], selected_rows[RECIPES[1]])]
    control = aggregate_rows(contrasts)
    screens = directional_screens(control)
    control.update(contrast="paired minus eight-seed mean training-derangement score",
                   directional_pairing_screens=screens, all_directional_pairing_screens_pass=all(screens.values()),
                   scientific_trial_count_increment=0, promotion_authorized=False,
                   interpretation="tests learning from trial-paired supervision; not absence of neural trial information")
    return panels, control
