"""Fitting-only LMP trajectory-repeatability weights and physical-label controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_lmp_high import _contrast
from primary_multichannel import ridge_solve
from primary_pca import prepare_split, projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_repeatability_weight_ridge_v1", "lmp_repeatability_weight_labels8_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))


def repeatability_statistics(fitting, fitting_directions):
    z, labels = np.asarray(fitting, dtype=float), np.asarray(fitting_directions)
    directions = sorted(set(labels.tolist()))
    if (z.ndim != 3 or z.shape[1] != 15 or z.shape[2] < 1 or labels.shape != (len(z),)
            or len(directions) != 7 or not set(directions) <= set(range(8))
            or not np.isfinite(z).all()
            or any(np.count_nonzero(labels == d) < 2 for d in directions)):
        raise ValueError("Finite complete fifteen-bin LMP and exactly seven fitting directions with >=2 trials required")
    total, paired, counts = [], [], []
    for direction in directions:
        block = z[labels == direction]
        count = len(block)
        energy = np.square(block).sum(axis=(0, 1))
        total.append(energy/(15*count))
        paired.append((np.square(block.sum(axis=0)).sum(axis=0)-energy)/(15*count*(count-1)))
        counts.append(count)
    t, p = np.mean(total, axis=0), np.mean(paired, axis=0)
    if not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(t < 0):
        raise ArithmeticError("Nonfinite or negative repeatability energy")
    raw = [float(p[e]/t[e]) if t[e] > 0 else None for e in range(z.shape[2])]
    if any(value is not None and not np.isfinite(value) for value in raw):
        raise ArithmeticError("Nonfinite positive-energy agreement ratio")
    r = np.array([0. if value is None else np.clip(value, 0., 1.) for value in raw])
    category = ["zero_energy" if value is None else "lower_clipped" if value < 0
                else "upper_clipped" if value > 1 else "unclipped" for value in raw]
    return {"T": t, "P": p, "raw_ratio": raw, "r": r, "a": .1+r,
            "clip_category": category, "fitting_directions": directions,
            "fitting_direction_counts": counts}


def trace_weights(base_weights, gram):
    a, g = np.asarray(base_weights, dtype=float), np.asarray(gram, dtype=float)
    if (a.ndim != 1 or len(a) < 1 or g.shape != (len(a), len(a))
            or not np.isfinite(a).all() or not np.isfinite(g).all()
            or np.any(a < .1) or np.any(a > 1.1) or np.any(np.diag(g) < 0)):
        raise ValueError("Fixed positive base weights and finite nonnegative-energy Gram required")
    diagonal = np.diag(g)
    active = diagonal > 0
    trace = float(diagonal.sum())
    denominator = float(a@diagonal)
    if trace == 0:
        if np.any(g != 0):
            raise ArithmeticError("Zero trace must have an exactly zero physical Gram")
        formula = np.ones(len(a))
        identity = True
    else:
        if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(trace):
            raise ArithmeticError("Positive finite trace-normalization denominator required")
        formula = np.sqrt(a*trace/denominator)
        identity = bool(np.all(a[active] == a[active][0]))
    # Exact algebraic identity on positive-energy columns; zero-input columns
    # have zero ridge coefficients, so all-one effective weights are safe.
    weights = np.ones(len(a)) if identity else formula
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ArithmeticError("Positive finite applied weights required")
    return weights, {"structural_identity": identity,
        "positive_energy_columns": np.flatnonzero(active).tolist(),
        "base_weights": a.tolist(), "weights": weights.tolist(), "formula_weights": formula.tolist(),
        "original_trace": trace, "normalization_denominator": denominator,
        "weighted_trace_from_diagonal": float(np.square(weights)@diagonal)}


def weight_map(electrode_ids, session, replicate):
    ids = np.asarray(electrode_ids)
    if (ids.ndim != 1 or len(ids) < 1 or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 1) or not np.array_equal(ids, np.unique(ids)) or replicate not in REPLICATES):
        raise ValueError("Canonical ascending unique positive physical IDs and a fixed replicate required")
    mapping = list(range(len(ids)))
    literal = f"ep05-lmp-repeatability-weight-label-v1|20261004|{session}|{replicate}"
    rng = random.Random()
    rng.seed(literal, version=2)
    rng.shuffle(mapping)
    mapping = np.asarray(mapping, dtype=int)
    return mapping, {"replicate": replicate, "seed_literal": literal,
                     "recipient_electrode_ids": ids.tolist(), "donor_electrode_ids": ids[mapping].tolist()}


def ridge_predictions(fitting, evaluation, residual, check=lambda: None):
    phi = fitting.reshape(-1, fitting.shape[-1])
    test = evaluation.reshape(-1, evaluation.shape[-1])
    output = []
    for penalty in LAMBDAS:
        check()
        output.append(test@ridge_solve(phi, residual, penalty))
    return np.asarray(output)


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed weight-label replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Weight-label support differs: {key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("Weight-control penalty identities differ")
        correct = float(np.mean([row["candidate_SSE"] for row in seeds]))
        wrong = float(np.mean([row["wrong_pair_SSE"] for row in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
                     "candidate_SSE": correct, "wrong_pair_SSE": wrong,
                     "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
                     "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
            "weight_label_replicate_ids": list(REPLICATES), "weight_label_Monte_Carlo_draws": 8,
            "weight_label_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]}
                                       for r, b in zip(REPLICATES, banks)],
            "weight_label_aggregation": "mean correct/wrong SSE; not averaged predictions"}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original current-bin LMP/XY/trial support required")
    if electrode_ids is None or len(electrode_ids) != x.shape[-1]:
        raise ValueError("Exact original physical electrode identities required")
    maps = [weight_map(electrode_ids, session, r) for r in REPLICATES]
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
        true_u = residual-intercept
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        z, state = prepare_split(x, train)
        statistics = repeatability_statistics(z[train], labels[train])
        flat = z[train].reshape(-1, x.shape[-1])
        gram = flat.T@flat/len(flat)
        active = np.diag(gram) > 0
        arms, covariances, equal_candidate = [], [], []
        candidate_weights = None
        for arm in range(1+len(REPLICATES)):
            check()
            a = statistics["a"] if arm == 0 else statistics["a"][maps[arm-1][0]]
            weights, info = trace_weights(a, gram)
            if arm == 0:
                candidate_weights = weights
            else:
                equal_candidate.append(bool(np.array_equal(weights[active], candidate_weights[active])))
            transformed = z*weights
            arms.append(info)
            covariances.append(projected_covariance(transformed[train]))
            fitted = ridge_predictions(transformed[train], transformed[test], true_u, check)
            for index in range(len(LAMBDAS)):
                value = base+fitted[index].reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(true_u),
            "physical_mean": state["physical_mean"].tolist(), "physical_sd": state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "repeatability": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in statistics.items()},
            "paired_weights": arms[0], "control_weights_by_replicate": arms[1:],
            "paired_weighted_covariance": covariances[0],
            "control_weighted_covariance_by_replicate": covariances[1:],
            "controls_equal_candidate_by_replicate": equal_candidate})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.update(physical_electrode_count=x.shape[-1], physical_electrode_ids=np.asarray(electrode_ids).tolist(),
        direction_fit_metadata=metadata, native_XY_scaled=False, positive_floor=.1,
        weights_use_velocity=False, post_weight_standardization=False,
        weight_label_maps=[details for _, details in maps],
        structural_identity=all(row["paired_weights"]["structural_identity"] for row in metadata),
        all_controls_equal_candidate=all(all(row["controls_equal_candidate_by_replicate"]) for row in metadata))
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    control = average_control_banks(controls)
    control.update(physical_normalization_preserved=True, true_native_targets=True,
        controls_have_own_Gram=True,
        structural_identity=all(arm["structural_identity"] for row in metadata for arm in row["control_weights_by_replicate"]),
        diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != len(SESSIONS)
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete sixteen-session candidate/control/reference banks required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete repeatability-weight/control recipes required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original lambda scores required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("Weight/control/reference baseline or target support differs")
        if banks[RECIPES[1]]["weight_label_replicate_ids"] != list(REPLICATES):
            raise ValueError("All eight fixed weight-label replicates required")
    candidate_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, anchor_matched = {r: [] for r in RECIPES}, []
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(candidate_banks, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                   "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                   "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seeds = bank["weight_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed-level scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            if recipe == RECIPES[0] and "direction_fit_metadata" in bank:
                def diagnostic(weight, covariance):
                    return {**weight, "trace": covariance["trace"], "numerical_rank": covariance["numerical_rank"],
                            "effective_df": covariance["ridge_df_by_lambda"][chosen]["effective_df"]}
                row["selected_weight_diagnostics"] = [{"held_direction": d["held_direction"],
                    "repeatability": d["repeatability"],
                    "paired": diagnostic(d["paired_weights"], d["paired_weighted_covariance"]),
                    "controls": [diagnostic(w, c) for w, c in zip(d["control_weights_by_replicate"], d["control_weighted_covariance_by_replicate"])],
                    "controls_equal_candidate_by_replicate": d["controls_equal_candidate_by_replicate"]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        anchor_matched.append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[held]["lambda_rows"][chosen]})
    identities = {recipe: all(sessions[s][recipe]["structural_identity"] for s in SESSIONS) for recipe in RECIPES}
    same = all(candidate_banks[s]["all_controls_equal_candidate"] for s in SESSIONS)
    charges = {RECIPES[0]: int(not identities[RECIPES[0]]), RECIPES[1]: int(not identities[RECIPES[1]] and not same)}
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      structural_identity=identities[recipe], scientific_trial_count_increment=charges[recipe],
                      promotion_authorized=False, search_complete=False, evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "Repeatability weighting minus mean weight-label control")
    contrasts = {"independently_selected": _contrast(selected[RECIPES[0]], anchor_panel["sessions"], "Repeatability weighting minus independently selected LMP"),
                 "repeatability_lambda_matched": _contrast(selected[RECIPES[0]], anchor_matched, "Repeatability weighting minus LMP at candidate-selected lambda")}
    screens = {k: c["all_directional_screens_pass"] for k, c in contrasts.items()}
    screens["weight_labels"] = control["all_directional_screens_pass"]
    return {"panels": panels, "weight_label_control": control, "anchor_contrasts": contrasts,
            "matched_lambda_anchor_sessions": anchor_matched, "weighting_screens": screens,
            "all_weighting_screens_pass": all(screens.values()), "structural_identity_by_recipe": identities,
            "all_controls_equal_candidate": same, "scientific_trial_count_by_recipe": charges,
            "scientific_trial_count_increment": sum(charges.values()),
            "control_trial_count_increment": charges[RECIPES[1]]}
