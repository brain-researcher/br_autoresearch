"""Direct native-XY sensitivity and within-direction training-target controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_lmp_high import _contrast
from primary_pca import projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_full_target_ridge_sensitivity_v1", "lmp_full_target_training_labels8_sensitivity_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))


def prepare_split(x, y, labels, held):
    train, test = labels != held, labels == held
    template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
    residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
    common_intercept = residual.mean(axis=0)
    baseline_training = template.predict(labels[train]*np.pi/4)+common_intercept
    baseline_evaluation = template.predict(labels[test]*np.pi/4)+common_intercept
    # Preserve the original full-channel anchor's flatten-then-reduce path.
    flat = x[train].reshape(-1, x.shape[-1])
    mean, sd = flat.mean(axis=0), flat.std(axis=0)
    scale = np.where(sd > 0, sd, 1.)
    mu = y[train].reshape(-1, 2).mean(axis=0)
    return {"train": train, "test": test, "physical_mean": mean, "physical_sd": sd,
        "fitting": (flat-mean)/scale,
        "evaluation": ((x[test]-mean)/scale).reshape(-1, x.shape[-1]),
        "mu": mu, "centered_targets": y[train].reshape(-1, 2)-mu,
        "baseline_training": baseline_training, "baseline_evaluation": baseline_evaluation,
        "baseline_equals_mu": bool(np.all(baseline_training == mu) and np.all(baseline_evaluation == mu))}


def trial_map(ids, labels, session, held, replicate):
    ids, labels = np.asarray(ids), np.asarray(labels)
    numeric = numeric_trial_ids(ids, session)
    allowed = sorted(set(range(8))-{held})
    if (held not in range(8) or replicate not in REPLICATES or labels.shape != ids.shape
            or set(labels.tolist()) != set(allowed)):
        raise ValueError("Original native IDs and exactly seven fitting directions required")
    mapping, metadata = np.empty(len(ids), dtype=int), []
    for direction in allowed:
        positions = sorted(np.flatnonzero(labels == direction).tolist(), key=lambda i: int(numeric[i]))
        donors = list(positions)
        literal = f"ep05-lmp-full-target-training-label-v1|20261004|{session}|{held}|{replicate}|{direction}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donors)
        mapping[positions] = donors
        metadata.append({"direction": direction, "seed_literal": literal,
                         "recipient_trial_ids": ids[positions].tolist(), "donor_trial_ids": ids[donors].tolist()})
    return mapping, metadata


def solve_path(gram, cross, nfit, check=lambda: None):
    g, c = np.asarray(gram, dtype=float), np.asarray(cross, dtype=float)
    if (g.ndim != 2 or g.shape[0] < 1 or g.shape[0] != g.shape[1] or c.shape != (len(g), 2)
            or type(nfit) is not int or nfit < 1 or not np.isfinite(g).all() or not np.isfinite(c).all()):
        raise ValueError("Finite complete physical Gram/native XY cross-product and fitting-row count required")
    path = []
    for penalty in LAMBDAS:
        check()
        beta = np.linalg.solve(g+nfit*penalty*np.eye(len(g)), c)
        if not np.isfinite(beta).all():
            raise ArithmeticError("Nonfinite direct ridge coefficients")
        path.append(beta)
    return np.asarray(path)


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed training-target label controls required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError("Training-label support differs: "+key)
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("Original five penalties required in every control")
        correct, wrong = (float(np.mean([row[key] for row in seeds])) for key in ("candidate_SSE", "wrong_pair_SSE"))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
        "training_label_replicate_ids": list(REPLICATES),
        "training_label_seed_rows": [{"replicate": r, "lambda_rows": bank["lambda_rows"]} for r, bank in zip(REPLICATES, banks)],
        "training_label_aggregation": "mean correct and exact wrong-pair SSE; not averaged predictions"}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1 or y.shape != (len(y), 15, 2)
            or labels.shape != (len(y),) or ids.shape != (len(y),) or set(labels.tolist()) != set(range(8))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original current-bin LMP/native XY/trial support required")
    electrodes = np.asarray(electrode_ids)
    if (electrodes.shape != (x.shape[-1],) or not np.issubdtype(electrodes.dtype, np.integer)
            or np.any(electrodes < 1) or not np.array_equal(electrodes, np.unique(electrodes))):
        raise ValueError("Original sorted unique positive physical electrode IDs required")
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    controls = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_split(x, y, labels, held)
        train, test, z = split["train"], split["test"], split["fitting"]
        baseline[test] = split["baseline_evaluation"]
        gram, nfit = z.T@z, len(z)
        v = split["centered_targets"]
        paired_cross = z.T@v
        crosses, maps, equal = [], [], []
        for arm in range(1+len(REPLICATES)):
            check()
            cross = paired_cross
            if arm:
                mapping, details = trial_map(ids[train], labels[train], session, held, arm-1)
                permuted = v.reshape(-1, 15, 2)[mapping].reshape(-1, 2)
                cross = z.T@permuted
                crosses.append(cross.tolist())
                maps.append({"replicate": arm-1, "directions": details})
                equal.append(bool(np.array_equal(cross, paired_cross)))
            coefficients = solve_path(gram, cross, nfit, check)
            for index, beta in enumerate(coefficients):
                prediction = split["mu"]+(split["evaluation"]@beta).reshape(-1, 15, 2)
                if arm:
                    controls[arm-1, index, test] = prediction
                else:
                    paired[index, test] = prediction
        metadata.append({"held_direction": held, "fitting_bin_rows": nfit,
            "physical_mean": split["physical_mean"].tolist(), "physical_sd": split["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(split["physical_sd"] == 0).tolist(),
            "mu": split["mu"].tolist(), "physical_Gram": gram.tolist(),
            "physical_covariance": projected_covariance(z), "paired_cross": paired_cross.tolist(),
            "control_cross_by_replicate": crosses, "training_target_maps": maps,
            "controls_equal_candidate_by_replicate": equal, "baseline_equals_mu": split["baseline_equals_mu"]})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    same = all(all(row["controls_equal_candidate_by_replicate"]) for row in metadata)
    candidate.update(physical_electrode_count=x.shape[-1], physical_electrode_ids=electrodes.tolist(),
        direction_fit_metadata=metadata, fitting_trial_ids=ids.tolist(),
        fitting_trial_directions=labels.tolist(), native_XY_scaled=False, baseline_added_to_prediction=False,
        prediction="true-training-global mu plus standardized-LMP ridge prediction of full XY minus mu",
        structural_residual_anchor_duplicate=all(row["baseline_equals_mu"] for row in metadata),
        all_controls_equal_candidate=same, sensitivity_only=True, promotion_authorized=False)
    control = average_control_banks([score_bank(y, labels, baseline, controls[r], RECIPES[1]) for r in REPLICATES])
    control.update(training_targets_permuted_as_whole_trajectories=True, within_direction_only=True,
        physical_Gram_shared_exactly=True, true_training_mu_shared_exactly=True, evaluation_unpermuted=True,
        structural_paired_duplicate=same, native_XY_scaled=False, baseline_added_to_prediction=False,
        sensitivity_only=True, promotion_authorized=False, diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: controls}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != 16
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Both complete sensitivity recipes and original residual-LMP anchor required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Complete candidate and control required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [row["lambda"] for row in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(row[k]) for row in bank["lambda_rows"]
                               for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Finite complete original lambda banks required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("Direct/residual/control baseline or target support differs")
        if (banks[RECIPES[1]]["training_label_replicate_ids"] != list(REPLICATES)
                or banks[RECIPES[1]]["structural_paired_duplicate"] != banks[RECIPES[0]]["all_controls_equal_candidate"]):
            raise ValueError("All eight training controls and consistent exact duplicate metadata required")
    paired = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, matched = {r: [] for r in RECIPES}, []
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(paired, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seeds = bank["training_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            elif "direction_fit_metadata" in bank:
                row["selected_fit_diagnostics"] = [{"held_direction": d["held_direction"], "mu": d["mu"],
                    "baseline_equals_mu": d["baseline_equals_mu"],
                    "controls_equal_candidate_by_replicate": d["controls_equal_candidate_by_replicate"],
                    "trace": d["physical_covariance"]["trace"], "numerical_rank": d["physical_covariance"]["numerical_rank"],
                    "effective_df": d["physical_covariance"]["ridge_df_by_lambda"][chosen]["effective_df"]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        matched.append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[held]["lambda_rows"][chosen]})
    duplicate = all(paired[s]["structural_residual_anchor_duplicate"] for s in SESSIONS)
    same = all(paired[s]["all_controls_equal_candidate"] for s in SESSIONS)
    charges = {RECIPES[0]: int(not duplicate), RECIPES[1]: int(not same)}
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      scientific_trial_count_increment=charges[recipe], sensitivity_only=True,
                      promotion_authorized=False, search_complete=False, evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "Full-target fit minus mean training-target label control")
    contrasts = {"independently_selected": _contrast(selected[RECIPES[0]], anchor_panel["sessions"], "Full-target minus independently selected residual LMP"),
                 "full_target_lambda_matched": _contrast(selected[RECIPES[0]], matched, "Full-target minus residual LMP at full-target-selected lambda")}
    return {"panels": panels, "training_target_label_control": control, "anchor_contrasts": contrasts,
        "matched_lambda_anchor_sessions": matched, "structural_residual_anchor_duplicate": duplicate,
        "all_controls_equal_candidate": same, "scientific_trial_count_by_recipe": charges,
        "scientific_trial_count_increment": sum(charges.values()), "control_trial_count_increment": charges[RECIPES[1]],
        "training_pairing_directional_screen_pass": control["all_directional_screens_pass"],
        "sensitivity_only": True, "promotion_authorized": False,
        "anchor_noninferiority_requirement": None}
