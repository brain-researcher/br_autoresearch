"""Fixed current-bin LMP/high ridge and eight matched high-trial controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_history import ridge_predictions
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_high_joint_ridge_v1", "lmp_high_trial_derangement8_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))
DERANGEMENT_DRAW_BOUND = 10000


def prepare_split(lmp, high, train):
    """All-trial physical-coordinate standardization using fitting rows only."""
    outputs = []
    for block in (lmp, high):
        block = np.asarray(block, dtype=float)
        mean, sd = block[train].mean(axis=(0, 1)), block[train].std(axis=(0, 1))
        outputs.append((block-mean)/np.where(sd == 0, 1., sd))
    return tuple(outputs)


def trial_maps(trial_ids, labels, session, replicate):
    """Return donor positions in input order, with canonical original-ID maps."""
    ids, labels = np.asarray(trial_ids), np.asarray(labels)
    if (replicate not in REPLICATES or ids.ndim != 1 or ids.shape != labels.shape
            or len(set(map(str, ids.tolist()))) != len(ids)
            or set(labels.tolist()) != set(range(8))):
        raise ValueError("Unique original IDs, eight directions and fixed replicate required")
    result, records = np.arange(len(ids)), []
    for direction in range(8):
        ordered = sorted(np.flatnonzero(labels == direction), key=lambda i: str(ids[i]))
        if len(ordered) < 2:
            raise ValueError("Each frozen direction needs at least two complete trials")
        literal = f"ep05-lmp-high-trial-label-v1|20261003|{session}|{direction}|{replicate}"
        rng = random.Random()
        rng.seed(literal, version=2)
        for draw in range(1, DERANGEMENT_DRAW_BOUND+1):
            donor = ordered.copy()
            rng.shuffle(donor)
            if all(a != b for a, b in zip(ordered, donor)):
                result[ordered] = donor
                records.append({"direction": direction, "replicate": replicate,
                    "seed_literal": literal, "draw_count": draw,
                    "recipient_trial_ids": ids[ordered].tolist(), "donor_trial_ids": ids[donor].tolist()})
                break
        else:
            raise RuntimeError("Fixed high-trial derangement bound exhausted; no reseed")
    return result, records


def joint_arm(z_lmp, z_high, train, donors=None):
    """LMP first, then whole high block; center fitting columns without scaling."""
    high = z_high if donors is None else z_high[donors]
    joint = np.concatenate((z_lmp, high), axis=-1)
    return joint-joint[train].mean(axis=(0, 1))


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed high-trial replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"High-trial replicate support differs: {key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [b["lambda_rows"][index] for b in banks]
        if any(r["lambda"] != penalty for r in seeds):
            raise ValueError("High-trial lambda identities differ")
        correct = float(np.mean([r["candidate_SSE"] for r in seeds]))
        wrong = float(np.mean([r["wrong_pair_SSE"] for r in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
        "high_trial_replicate_ids": list(REPLICATES), "high_trial_Monte_Carlo_draws": 8,
        "high_trial_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]}
                                  for r, b in zip(REPLICATES, banks)],
        "high_trial_aggregation": "mean correct/wrong SSE; not averaged predictions",
        "lambda_selection": "joint-selected outer-training lambda shared by all eight control refits"}


def fit_banks(lmp, high, targets, directions, trial_ids, session, check=lambda: None):
    lmp, high, y = (np.asarray(x, dtype=float) for x in (lmp, high, targets))
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    if (lmp.ndim != 3 or lmp.shape[:2] != (len(y), 15) or lmp.shape[-1] < 1
            or high.shape != (*lmp.shape[:2], 2*lmp.shape[-1]) or y.shape != (len(y), 15, 2)
            or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not all(np.isfinite(x).all() for x in (lmp, high, y))
            or set(labels.tolist()) != set(range(8))):
        raise ValueError("Complete matching LMP/high/XY physical-coordinate support required")
    mappings = [trial_maps(ids, labels, session, r) for r in REPLICATES]
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    shuffled = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        train, test = labels != held, labels == held
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        centered = residual-intercept
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        zl, zh = prepare_split(lmp, high, train)
        for arm in range(1+len(REPLICATES)):
            check()
            donors = None if arm == 0 else mappings[arm-1][0]
            # Directions are disjoint train/evaluation partitions; maps stay inside them.
            if donors is not None and not np.array_equal(train[donors], train):
                raise ArithmeticError("High donor crossed fitting/evaluation boundary")
            joint = joint_arm(zl, zh, train, donors)
            fitted = ridge_predictions(joint[train], joint[test], centered, check)
            for index in range(len(LAMBDAS)):
                prediction = base+fitted[index].reshape(base.shape)
                if arm == 0:
                    paired[index, test] = prediction
                else:
                    shuffled[arm-1, index, test] = prediction
        metadata.append({"held_direction": held, "fitting_bin_rows": len(centered),
            "LMP_zero_SD_columns": np.flatnonzero(lmp[train].std(axis=(0, 1)) == 0).tolist(),
            "high_zero_SD_columns": np.flatnonzero(high[train].std(axis=(0, 1)) == 0).tolist()})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.update(feature_width=3*lmp.shape[-1], LMP_electrodes=lmp.shape[-1],
        high_bands_hz=[[100, 200], [200, 400]], direction_fit_metadata=metadata,
        column_order="LMP physical electrode; high physical electrode then100-200,200-400Hz",
        high_trial_maps=[{"replicate": r, "directions": records}
                         for r, (_, records) in zip(REPLICATES, mappings)],
        normalization="training physical-coordinate population mean/SD; zeroSD->1; no block rescaling")
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    control = average_control_banks(controls)
    control.update(feature_width=3*lmp.shape[-1], high_trial_map_metadata="joint bank high_trial_maps",
        LMP_inputs_unchanged=True, evaluation_targets_unchanged=True,
        high_block_Gram_preserved=True, cross_block_Gram_preserved=False)
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def _contrast(left, right, label):
    right = {r["session_id"]: r for r in right}
    rows = [{"session_id": a["session_id"], "animal": a["animal"], "lambda": a["lambda"],
        "reference_lambda": right[a["session_id"]]["lambda"],
        "delta_R2": a["delta_R2"]-right[a["session_id"]]["delta_R2"],
        "identity_increment": a["identity_increment"]-right[a["session_id"]]["identity_increment"]}
        for a in left]
    summary = aggregate_rows(rows)
    screens = directional_screens(summary)
    return {**summary, "contrast": label, "directional_screens": screens,
            "all_directional_screens_pass": all(screens.values()),
            "scientific_trial_count_increment": 0, "promotion_authorized": False}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)
            or len(anchor_panel["sessions"]) != len(SESSIONS)):
        raise ValueError("Both complete joint banks and preserved sixteen-session LMP reference required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both joint/control recipes required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite fixed-lambda score rows required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("Joint/control/reference scoring support differs")
        if banks[RECIPES[1]]["high_trial_replicate_ids"] != list(REPLICATES):
            raise ValueError("All high trial-control replicates required")
    joint_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    chosen_rows, anchor_matched = {r: [] for r in RECIPES}, []
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(joint_banks, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                "outer_training_sessions": training, "joint_inner_animal_balanced_scores": scores,
                "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seed_rows = bank["high_trial_seed_rows"]
                if [r["replicate"] for r in seed_rows] != list(REPLICATES):
                    raise ValueError("Complete ordered seed scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]}
                                                for r in seed_rows]
            chosen_rows[recipe].append(row)
        anchor_matched.append({"session_id": held, "animal": ANIMALS[held],
                               **anchor_banks[held]["lambda_rows"][chosen]})
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(chosen_rows[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005,
                   **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
            scientific_trial_count_increment=1, promotion_authorized=False, search_complete=False,
            role="candidate" if recipe == RECIPES[0] else "matched high-block trial control",
            evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(chosen_rows[RECIPES[0]], chosen_rows[RECIPES[1]], "joint minus mean high-trial control")
    contrasts = {"independently_selected": _contrast(chosen_rows[RECIPES[0]], anchor_panel["sessions"],
                    "joint minus independently selected preserved LMP"),
                 "joint_lambda_matched": _contrast(chosen_rows[RECIPES[0]], anchor_matched,
                    "joint minus preserved LMP at joint-selected lambda")}
    complement = {name: c["all_directional_screens_pass"] for name, c in contrasts.items()}
    complement["trial_alignment"] = control["all_directional_screens_pass"]
    return {"panels": panels, "trial_alignment_control": control, "anchor_contrasts": contrasts,
        "matched_lambda_anchor_sessions": anchor_matched, "complementarity_screens": complement,
        "all_complementarity_screens_pass": all(complement.values())}
