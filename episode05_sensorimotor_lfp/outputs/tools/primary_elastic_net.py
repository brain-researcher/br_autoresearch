"""Fixed native-XY elastic net and whole-training-trial correspondence controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_lmp_high import _contrast
from primary_pca import prepare_split
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_full_channel_elastic_net_v1", "lmp_full_channel_elastic_net_training_labels8_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))
MAX_SWEEPS = 10000


def penalty_scale(residual):
    u = np.asarray(residual, dtype=float)
    if u.ndim not in (2, 3) or u.shape[-1] != 2 or u.size == 0 or not np.isfinite(u).all():
        raise ValueError("Finite nonempty native XY fitting residuals required")
    scale = float(np.sqrt(np.mean(np.square(u))))
    if not np.isfinite(scale):
        raise ArithmeticError("Nonfinite common fitting velocity scale")
    return scale


def kkt_certificate(gram, cross, coefficients, penalty, scale):
    g = gram@coefficients-cross+penalty*coefficients
    threshold = penalty*scale
    residual = np.where(coefficients != 0,
                        np.abs(g+threshold*np.sign(coefficients)),
                        np.maximum(np.abs(g)-threshold, 0.))
    tolerance = 1e-8*max(1., float(np.max(np.abs(cross))))
    maximum, norm = float(np.max(residual)), float(np.linalg.norm(residual))
    if not np.isfinite(residual).all():
        raise ArithmeticError("Nonfinite exact-gradient KKT certificate")
    return {"maximum_KKT_residual": maximum, "KKT_residual_Frobenius_norm": norm,
            "KKT_tolerance": tolerance, "converged": maximum <= tolerance,
            "objective_error_upper_bound": norm*norm/(2*penalty)}


def solve_path(gram, cross, scale, check=lambda: None, max_sweeps=MAX_SWEEPS):
    """Cyclic physical-coordinate native-XY solves; fixed descending warm starts."""
    g, c = np.asarray(gram, dtype=float), np.asarray(cross, dtype=float)
    if (g.ndim != 2 or g.shape[0] < 1 or g.shape[0] != g.shape[1]
            or c.shape != (len(g), 2) or not np.isfinite(g).all() or not np.isfinite(c).all()
            or not np.isfinite(scale) or scale < 0 or type(max_sweeps) is not int
            or not 1 <= max_sweeps <= MAX_SWEEPS or np.any(np.diag(g) < 0)
            or not np.allclose(g, g.T, rtol=1e-13, atol=1e-13)):
        raise ValueError("Finite symmetric physical Gram, native XY covariance and bounded sweep count required")
    coefficients = np.zeros_like(c)
    path = np.empty((len(LAMBDAS), *c.shape))
    diagnostics = [None]*len(LAMBDAS)
    for index in reversed(range(len(LAMBDAS))):
        penalty = LAMBDAS[index]
        threshold = penalty*scale
        for sweep in range(1, max_sweeps+1):
            check()
            for electrode in range(len(g)):
                # The two response updates are independent with identical G.
                partial = c[electrode]-g[electrode]@coefficients+g[electrode, electrode]*coefficients[electrode]
                coefficients[electrode] = (np.sign(partial)*np.maximum(np.abs(partial)-threshold, 0.)
                                             /(g[electrode, electrode]+penalty))
            if not np.isfinite(coefficients).all():
                raise ArithmeticError("Nonfinite elastic-net coefficients")
            certificate = kkt_certificate(g, c, coefficients, penalty, scale)
            if certificate["converged"]:
                active = coefficients != 0
                diagnostics[index] = {"lambda": penalty, "completed_sweeps": sweep,
                    "maximum_sweeps": MAX_SWEEPS, "penalty_scale": float(scale),
                    "L1_penalty_coefficient": threshold, "L2_penalty_coefficient": penalty,
                    "zero_coefficients": int(np.count_nonzero(~active)),
                    "active_X": int(np.count_nonzero(active[:, 0])),
                    "active_Y": int(np.count_nonzero(active[:, 1])),
                    "active_channel_union": int(np.count_nonzero(np.any(active, axis=1))),
                    "active_channel_intersection": int(np.count_nonzero(np.all(active, axis=1))),
                    "coefficient_L1_norm": float(np.abs(coefficients).sum()),
                    "coefficient_Frobenius_norm": float(np.linalg.norm(coefficients)),
                    **certificate}
                path[index] = coefficients
                break
        else:
            raise RuntimeError(f"Elastic-net lambda={penalty} did not satisfy KKT certificate within {max_sweeps} sweeps")
    return path, diagnostics


def trial_map(ids, labels, session, held, replicate):
    ids, labels = np.asarray(ids), np.asarray(labels)
    numeric = numeric_trial_ids(ids, session)
    allowed = sorted(set(range(8))-{held})
    if (held not in range(8) or replicate not in REPLICATES or labels.shape != ids.shape
            or set(labels.tolist()) != set(allowed)):
        raise ValueError("Unique original IDs and exactly seven fitting directions required")
    mapping, metadata = np.empty(len(ids), dtype=int), []
    for direction in allowed:
        positions = sorted(np.flatnonzero(labels == direction).tolist(), key=lambda i: int(numeric[i]))
        donors = list(positions)
        literal = f"ep05-lmp-elastic-net-training-label-v1|20261004|{session}|{held}|{replicate}|{direction}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donors)
        mapping[positions] = donors
        metadata.append({"direction": direction, "seed_literal": literal,
                         "recipient_trial_ids": ids[positions].tolist(), "donor_trial_ids": ids[donors].tolist()})
    return mapping, metadata


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed training-label replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Training-control support differs: {key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("Training-control penalty identities differ")
        correct = float(np.mean([row["candidate_SSE"] for row in seeds]))
        wrong = float(np.mean([row["wrong_pair_SSE"] for row in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
                     "candidate_SSE": correct, "wrong_pair_SSE": wrong,
                     "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
                     "identity_increment": (wrong-correct)/first["denominator"]})
    result = {**first, "lambda_rows": rows, "elastic_net_two_response_fits": 8*8*len(LAMBDAS),
              "training_label_replicate_ids": list(REPLICATES), "training_label_Monte_Carlo_draws": 8,
              "training_label_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]}
                                            for r, b in zip(REPLICATES, banks)],
              "training_label_aggregation": "mean correct/wrong SSE; not averaged predictions"}
    result.pop("ridge_fits", None)
    return result


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original current-bin LMP/XY/trial support required")
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
        fit_z = z[train].reshape(-1, x.shape[-1])
        eval_z = z[test].reshape(-1, x.shape[-1])
        gram = fit_z.T@fit_z/len(fit_z)
        scale = penalty_scale(true_u)
        diagnostics, maps = [], []
        for arm in range(1+len(REPLICATES)):
            check()
            arm_z = fit_z
            if arm:
                mapping, details = trial_map(ids[train], labels[train], session, held, arm-1)
                arm_z = z[train][mapping].reshape(-1, x.shape[-1])
                maps.append({"replicate": arm-1, "directions": details})
            # Exact same physical Gram; only training correspondence changes C.
            cross = arm_z.T@true_u/len(true_u)
            coefficients, fit_info = solve_path(gram, cross, scale, check)
            diagnostics.append(fit_info)
            for index in range(len(LAMBDAS)):
                value = base+(eval_z@coefficients[index]).reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(true_u),
            "physical_mean": state["physical_mean"].tolist(), "physical_sd": state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "common_velocity_penalty_scale": scale, "physical_Gram_shared_exactly": True,
            "paired_fit_diagnostics_by_lambda": diagnostics[0],
            "control_fit_diagnostics_by_replicate_lambda": diagnostics[1:],
            "training_trial_maps_by_replicate": maps})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.pop("ridge_fits", None)
    candidate.update(physical_electrode_count=x.shape[-1], direction_fit_metadata=metadata,
        elastic_net_two_response_fits=8*len(LAMBDAS), common_velocity_penalty_scale=True,
        native_XY_scaled=False, coordinatewise_L1=True, maximum_sweeps=MAX_SWEEPS,
        all_folds_zero_residual_scale=all(row["common_velocity_penalty_scale"] == 0 for row in metadata))
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    control = average_control_banks(controls)
    control.update(training_LMP_permuted_as_whole_trials=True, physical_normalization_preserved=True,
                   fitting_targets_unpermuted=True, physical_Gram_shared_exactly=True,
                   diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != len(SESSIONS)
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete sixteen-session candidate/control/reference banks required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete elastic-net/training-control recipes required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original lambda scores required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("Elastic-net/control/reference baseline or target support differs")
        if banks[RECIPES[1]]["training_label_replicate_ids"] != list(REPLICATES):
            raise ValueError("All eight fixed training-label replicates required")
    paired_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, anchor_matched = {r: [] for r in RECIPES}, []
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(paired_banks, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                   "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                   "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seeds = bank["training_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed-level scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            if recipe == RECIPES[0] and "direction_fit_metadata" in bank:
                row["selected_fit_diagnostics"] = [{"held_direction": d["held_direction"],
                    "common_velocity_penalty_scale": d["common_velocity_penalty_scale"],
                    "paired": d["paired_fit_diagnostics_by_lambda"][chosen],
                    "controls": [c[chosen] for c in d["control_fit_diagnostics_by_replicate_lambda"]]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        anchor_matched.append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[held]["lambda_rows"][chosen]})
    # Only a structural zero residual everywhere removes the new model opportunity.
    # A sparse/all-zero optimum at positive residual scale remains a valid new negative result.
    duplicate = all(paired_banks[s]["all_folds_zero_residual_scale"] for s in SESSIONS)
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      scientific_trial_count_increment=0 if duplicate else 1, promotion_authorized=False,
                      search_complete=False, evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "Elastic net minus mean training-label control")
    contrasts = {"independently_selected": _contrast(selected[RECIPES[0]], anchor_panel["sessions"], "Elastic net minus independently selected LMP"),
                 "elastic_net_lambda_matched": _contrast(selected[RECIPES[0]], anchor_matched, "Elastic net minus LMP at candidate-selected lambda")}
    screens = {k: c["all_directional_screens_pass"] for k, c in contrasts.items()}
    screens["training_labels"] = control["all_directional_screens_pass"]
    return {"panels": panels, "training_label_control": control, "anchor_contrasts": contrasts,
            "matched_lambda_anchor_sessions": anchor_matched, "regularization_screens": screens,
            "all_regularization_screens_pass": all(screens.values()), "structural_duplicate": duplicate,
            "structural_baseline_duplicate": duplicate, "scientific_trial_count_increment": 0 if duplicate else 2}
