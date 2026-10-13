"""Training-only90%-variance PCA ridge and frozen loading-label controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_history import ridge_predictions
from primary_lmp_high import _contrast
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_pca90_ridge_v1", "lmp_pca90_loading_labels8_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))


def prepare_split(x, train):
    mean, sd = x[train].mean(axis=(0, 1)), x[train].std(axis=(0, 1))
    return (x-mean)/np.where(sd == 0, 1., sd), {"physical_mean": mean, "physical_sd": sd}


def _covariance(fitting):
    flat = fitting.reshape(-1, fitting.shape[-1])
    covariance = flat.T@flat/len(flat)
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    tolerance = 64*np.finfo(np.float64).eps*len(values)*max(1., float(np.max(np.abs(values))))
    if not np.isfinite(values).all() or np.min(values) < -tolerance:
        raise ArithmeticError("Materially negative or nonfinite fitting covariance eigenvalue")
    return np.maximum(values[::-1], 0.), vectors[:, ::-1], tolerance, float(np.trace(covariance))


def pca_basis(fitting):
    values, vectors, tolerance, trace = _covariance(np.asarray(fitting, dtype=float))
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Positive fitting covariance required; no rank fallback")
    count = int(np.searchsorted(np.cumsum(values), .9*total, side="left"))+1
    while count < len(values) and values[count] == values[count-1]:
        count += 1
    return vectors[:, :count], {"eigenvalues": values.tolist(), "retained_components": count,
        "retained_variance_fraction": float(values[:count].sum()/total),
        "covariance_tolerance": tolerance, "trace": trace,
        "full_span": count == len(values), "whitened": False}


def loading_map(electrodes, session, replicate):
    if type(electrodes) is not int or electrodes < 1 or replicate not in REPLICATES:
        raise ValueError("Original electrode count and fixed replicate required")
    order = list(range(electrodes))
    rng = random.Random()
    rng.seed(f"ep05-lmp-pca90-loading-label-v1|20261004|{session}|{replicate}", version=2)
    rng.shuffle(order)
    return np.asarray(order, dtype=int)


def project_arm(z, basis, train, mapping=None):
    if mapping is not None:
        if (mapping.shape != (z.shape[-1],) or not np.issubdtype(mapping.dtype, np.integer)
                or not np.array_equal(np.sort(mapping), np.arange(z.shape[-1]))):
            raise ValueError("Complete physical loading-row permutation required")
        basis = basis[mapping]
    projected = z@basis
    return projected-projected[train].mean(axis=(0, 1))


def projected_covariance(features):
    values, _, tolerance, trace = _covariance(features)
    return {"eigenvalues": values.tolist(), "trace": trace,
        "numerical_rank": int(np.count_nonzero(values > tolerance)), "numerical_rank_tolerance": tolerance,
        "ridge_df_by_lambda": [{"lambda": p, "effective_df": float(np.sum(values/(values+p)))} for p in LAMBDAS],
        "diagnostic_only": True}


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed loading-label replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Loading-control support differs:{key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [b["lambda_rows"][index] for b in banks]
        if any(r["lambda"] != penalty for r in seeds):
            raise ValueError("Loading-control lambda identities differ")
        correct = float(np.mean([r["candidate_SSE"] for r in seeds]))
        wrong = float(np.mean([r["wrong_pair_SSE"] for r in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
        "loading_label_replicate_ids": list(REPLICATES), "loading_label_Monte_Carlo_draws": 8,
        "loading_label_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]} for r, b in zip(REPLICATES, banks)],
        "loading_label_aggregation": "mean correct/wrong SSE; not averaged predictions"}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1 or y.shape != (len(y), 15, 2)
            or labels.shape != (len(y),) or ids.shape != (len(y),) or len(set(ids.tolist())) != len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original current-bin LMP/XY/trial support required")
    maps = [loading_map(x.shape[-1], session, r) for r in REPLICATES]
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
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        z, state = prepare_split(x, train)
        basis, pca = pca_basis(z[train])
        diagnostics = []
        for arm in range(1+len(REPLICATES)):
            check()
            projected = project_arm(z, basis, train, None if arm == 0 else maps[arm-1])
            diagnostic = projected_covariance(projected[train])
            diagnostics.append(diagnostic)
            # Zero-energy controls are retained: positive-lambda ridge returns zero correction.
            prediction = ridge_predictions(projected[train], projected[test], residual-intercept, check)
            for index in range(len(LAMBDAS)):
                value = base+prediction[index].reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(residual), "pca": pca,
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "paired_projected_covariance": diagnostics[0], "control_projected_covariance_by_replicate": diagnostics[1:]})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.update(physical_electrode_count=x.shape[-1], retained_variance_target=.9, direction_fit_metadata=metadata,
        loading_maps=[{"replicate": r, "source_channel_positions": m.tolist(),
            "seed_literal": f"ep05-lmp-pca90-loading-label-v1|20261004|{session}|{r}"} for r, m in zip(REPLICATES, maps)],
        loading_map_basis="destination-to-source zero-based positions in source.common_LMP_electrodes",
        all_folds_full_span=all(r["pca"]["full_span"] for r in metadata), whitened=False)
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    control = average_control_banks(controls)
    control.update(PCA_refitted_after_permutation=False, physical_normalization_preserved=True,
                   diagnostics_location="candidate direction_fit_metadata; nominal K may exceed effective rank")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != len(SESSIONS)
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete sixteen-session candidate/control/reference banks required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete PCA/loading-control recipes required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original lambda scores required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("PCA/control/reference baseline or target support differs")
        if banks[RECIPES[1]]["loading_label_replicate_ids"] != list(REPLICATES):
            raise ValueError("All fixed loading-label replicates required")
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
                seeds = bank["loading_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed-level scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            if recipe == RECIPES[0] and "direction_fit_metadata" in bank:
                def diagnostic(d):
                    return {"trace": d["trace"], "numerical_rank": d["numerical_rank"],
                            "effective_df": d["ridge_df_by_lambda"][chosen]["effective_df"]}
                row["selected_projection_diagnostics"] = [{"held_direction": d["held_direction"],
                    "retained_components": d["pca"]["retained_components"],
                    "paired": diagnostic(d["paired_projected_covariance"]),
                    "controls": [diagnostic(c) for c in d["control_projected_covariance_by_replicate"]]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        anchor_matched.append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[held]["lambda_rows"][chosen]})
    duplicate = all(paired_banks[s]["all_folds_full_span"] for s in SESSIONS)
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
            scientific_trial_count_increment=0 if duplicate else 1, promotion_authorized=False, search_complete=False,
            evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "PCA minus mean frozen loading-label control")
    contrasts = {"independently_selected": _contrast(selected[RECIPES[0]], anchor_panel["sessions"], "PCA minus independently selected LMP"),
                 "pca_lambda_matched": _contrast(selected[RECIPES[0]], anchor_matched, "PCA minus LMP at PCA-selected lambda")}
    screens = {k: c["all_directional_screens_pass"] for k, c in contrasts.items()}
    screens["loading_labels"] = control["all_directional_screens_pass"]
    return {"panels": panels, "loading_label_control": control, "anchor_contrasts": contrasts,
        "matched_lambda_anchor_sessions": anchor_matched, "denoising_screens": screens,
        "all_denoising_screens_pass": all(screens.values()), "structural_full_span_duplicate": duplicate,
        "scientific_trial_count_increment": 0 if duplicate else 2}
