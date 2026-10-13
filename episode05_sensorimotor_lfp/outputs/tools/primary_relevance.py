"""Nested held-direction single-channel relevance, fixed top16 and rank-label controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_history import ridge_predictions
from primary_lmp_high import _contrast
from primary_pca import prepare_split, projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_inner_lodo_top16_ridge_v1", "lmp_inner_lodo_top16_rank_labels8_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))
TOP_K = 16
RANKING_LAMBDA = 1.


def single_channel_predictions(fitting, evaluation, centered_residual):
    """Independent scalar-input/two-output mean-loss ridge at fixed lambda1."""
    phi = fitting.reshape(-1, fitting.shape[-1])
    test = evaluation.reshape(-1, evaluation.shape[-1])
    residual = np.asarray(centered_residual, dtype=float)
    if residual.shape != (len(phi), 2):
        raise ValueError("Two native centered residual coordinates required")
    coefficient = (phi.T@residual)/(np.sum(phi*phi, axis=0)[:, None]+len(phi)*RANKING_LAMBDA)
    return test[:, :, None]*coefficient[None, :, :]


def rank_channels(features, targets, directions, channel_ids, held, check=lambda: None):
    labels, channels = np.asarray(directions), np.asarray(channel_ids)
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    if (held not in range(8) or set(labels.tolist()) != set(range(8))
            or channels.shape != (x.shape[-1],) or len(set(channels.tolist())) != len(channels)):
        raise ValueError("Original eight directions and unique physical IDs required")
    # Remove the outer evaluation rows before any inner numerical operation.
    allowed = labels != held
    x, y, labels = x[allowed], y[allowed], labels[allowed]
    total_gain, inner_rows = np.zeros(len(channels)), []
    for inner in sorted(set(labels.tolist())):
        check()
        train, test = labels != inner, labels == inner
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        baseline = (template.predict(labels[test]*np.pi/4)+intercept).reshape(-1, 2)
        z, _ = prepare_split(x, train)
        prediction = baseline[:, None, :]+single_channel_predictions(z[train], z[test], residual-intercept)
        actual = y[test].reshape(-1, 2)
        baseline_sse = float(np.sum((actual-baseline)**2))
        candidate_sse = np.sum((actual[:, None, :]-prediction)**2, axis=(0, 2))
        gain = baseline_sse-candidate_sse
        total_gain += gain
        inner_rows.append({"inner_held_direction": int(inner),
            "fitting_directions": sorted(set(labels[train].tolist())),
            "fitting_bin_rows": int(np.count_nonzero(train)*15),
            "evaluation_bin_rows": len(actual), "baseline_SSE": baseline_sse,
            "single_channel_SSE": candidate_sse.tolist(), "gain_SSE_by_physical_channel": gain.tolist()})
    common_rows = 15*len(y)
    gains = total_gain/common_rows
    if not np.isfinite(gains).all():
        raise ArithmeticError("Finite signed training-only ranking gains required")
    order = np.lexsort((channels, -gains))
    k = min(TOP_K, len(channels))
    cutoff = gains[order[k-1]]
    return order, {"ranking_lambda": RANKING_LAMBDA, "common_evaluation_bin_rows": common_rows,
        "physical_channel_ids": channels.tolist(), "signed_mean_gain_by_physical_channel": gains.tolist(),
        "strict_ranked_channel_ids": channels[order].tolist(), "strict_order_positions": order.tolist(),
        "positive_gain_channels": int(np.count_nonzero(gains > 0)),
        "raw_gain_range": [float(gains.min()), float(gains.max())],
        "all_raw_gains_tied": bool(np.all(gains == gains[0])),
        "cutoff_score": float(cutoff), "cutoff_tied_channels": int(np.count_nonzero(gains == cutoff)),
        "inner_fits": inner_rows}


def rank_map(electrodes, session, replicate):
    if type(electrodes) is not int or electrodes < 1 or replicate not in REPLICATES:
        raise ValueError("Original electrode count and fixed replicate required")
    order = list(range(electrodes))
    rng = random.Random()
    rng.seed(f"ep05-lmp-inner-lodo-top16-rank-label-v1|20261004|{session}|{replicate}", version=2)
    rng.shuffle(order)
    return np.asarray(order, dtype=int)


def retained_order(strict_order, mapping=None):
    order = np.asarray(strict_order)
    if (order.ndim != 1 or len(order) == 0 or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(len(order)))):
        raise ValueError("Strict ranking must cover each physical position once")
    if mapping is not None:
        mapping = np.asarray(mapping)
        if (mapping.shape != order.shape or not np.issubdtype(mapping.dtype, np.integer)
                or not np.array_equal(np.sort(mapping), np.arange(len(order)))):
            raise ValueError("Complete destination-to-source rank-label map required")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        order = np.argsort(rank[mapping])
    return order[:min(TOP_K, len(order))]


def retain_arm(z, order, train, mapping=None):
    selected = retained_order(order, mapping)
    features = z[:, :, selected]
    return features-features[train].mean(axis=(0, 1)), selected


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight rank-label control replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Rank-control support differs:{key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [b["lambda_rows"][index] for b in banks]
        if any(r["lambda"] != penalty for r in seeds):
            raise ValueError("Fixed control lambda identities differ")
        correct = float(np.mean([r["candidate_SSE"] for r in seeds]))
        wrong = float(np.mean([r["wrong_pair_SSE"] for r in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
        "rank_label_replicate_ids": list(REPLICATES), "rank_label_Monte_Carlo_draws": 8,
        "rank_label_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]} for r, b in zip(REPLICATES, banks)],
        "rank_label_aggregation": "mean correct/wrong SSE; not averaged predictions"}


def fit_banks(features, targets, directions, trial_ids, channel_ids, session, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids, channels = np.asarray(directions), np.asarray(trial_ids), np.asarray(channel_ids)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1 or y.shape != (len(y), 15, 2)
            or labels.shape != (len(y),) or ids.shape != (len(y),) or len(set(ids.tolist())) != len(y)
            or channels.shape != (x.shape[-1],) or not np.issubdtype(channels.dtype, np.integer)
            or channels.tolist() != sorted(set(channels.tolist())) or np.any(channels < 1)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original current-bin LMP/XY/trial/physical support required")
    maps = [rank_map(x.shape[-1], session, r) for r in REPLICATES]
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    shuffled = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        train, test = labels != held, labels == held
        order, ranking = rank_channels(x, y, labels, channels, held, check)
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        z, state = prepare_split(x, train)
        diagnostics, selections = [], []
        for arm in range(1+len(REPLICATES)):
            check()
            retained, selected = retain_arm(z, order, train, None if arm == 0 else maps[arm-1])
            diagnostics.append(projected_covariance(retained[train]))
            selections.append(channels[selected].tolist())
            prediction = ridge_predictions(retained[train], retained[test], residual-intercept, check)
            for index in range(len(LAMBDAS)):
                value = base+prediction[index].reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(residual), "ranking": ranking,
            "retained_channels": min(TOP_K, len(channels)), "selected_channel_ids": selections[0],
            "control_selected_channel_ids_by_replicate": selections[1:],
            "control_overlap_with_candidate": [len(set(selections[0]).intersection(s)) for s in selections[1:]],
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "paired_projected_covariance": diagnostics[0], "control_projected_covariance_by_replicate": diagnostics[1:]})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.update(physical_electrode_count=len(channels), fixed_top_k=TOP_K,
        ranking_lambda=RANKING_LAMBDA, direction_fit_metadata=metadata,
        nested_inner_scalar_channel_fits=8*7*len(channels), nested_inner_direction_fits=56,
        rank_maps=[{"replicate": r, "source_channel_positions": m.tolist(),
            "source_channel_ids": channels[m].tolist(),
            "seed_literal": f"ep05-lmp-inner-lodo-top16-rank-label-v1|20261004|{session}|{r}"}
            for r, m in zip(REPLICATES, maps)],
        all_folds_full_retention=len(channels) <= TOP_K)
    control = average_control_banks([score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES])
    control.update(ranking_refitted_after_permutation=False, physical_normalization_preserved=True,
                   diagnostics_location="candidate direction_fit_metadata; equal K need not imply equal covariance")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != len(SESSIONS)
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete sixteen-session candidate/control/reference banks required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete relevance/rank-control recipes required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original lambda scores required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("Relevance/control/reference baseline or target support differs")
        if banks[RECIPES[1]]["rank_label_replicate_ids"] != list(REPLICATES):
            raise ValueError("All fixed rank-label replicates required")
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
                seeds = bank["rank_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed-level scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            if recipe == RECIPES[0] and "direction_fit_metadata" in bank:
                def diagnostic(d):
                    return {"trace": d["trace"], "numerical_rank": d["numerical_rank"],
                            "effective_df": d["ridge_df_by_lambda"][chosen]["effective_df"]}
                row["selected_retention_diagnostics"] = [{"held_direction": d["held_direction"],
                    "retained_channels": d["retained_channels"], "selected_channel_ids": d["selected_channel_ids"],
                    "ranking_positive_channels": d["ranking"]["positive_gain_channels"],
                    "ranking_raw_gain_range": d["ranking"]["raw_gain_range"],
                    "ranking_all_raw_gains_tied": d["ranking"]["all_raw_gains_tied"],
                    "ranking_cutoff_tied_channels": d["ranking"]["cutoff_tied_channels"],
                    "control_overlap_with_candidate": d["control_overlap_with_candidate"],
                    "paired": diagnostic(d["paired_projected_covariance"]),
                    "controls": [diagnostic(c) for c in d["control_projected_covariance_by_replicate"]]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        anchor_matched.append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[held]["lambda_rows"][chosen]})
    duplicate = all(paired_banks[s]["all_folds_full_retention"] for s in SESSIONS)
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
            scientific_trial_count_increment=0 if duplicate else 1, promotion_authorized=False, search_complete=False,
            evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "relevance top16 minus mean rank-label control")
    contrasts = {"independently_selected": _contrast(selected[RECIPES[0]], anchor_panel["sessions"], "relevance top16 minus independently selected LMP"),
                 "relevance_lambda_matched": _contrast(selected[RECIPES[0]], anchor_matched, "relevance top16 minus LMP at relevance-selected lambda")}
    screens = {k: c["all_directional_screens_pass"] for k, c in contrasts.items()}
    screens["rank_labels"] = control["all_directional_screens_pass"]
    return {"panels": panels, "rank_label_control": control, "anchor_contrasts": contrasts,
        "matched_lambda_anchor_sessions": anchor_matched, "relevance_screens": screens,
        "all_relevance_screens_pass": all(screens.values()), "structural_full_retention_duplicate": duplicate,
        "scientific_trial_count_increment": 0 if duplicate else 2}
