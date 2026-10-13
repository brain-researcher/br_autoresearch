"""Fixed native-target covariance-SVD input projection and basis-only controls."""
import random

import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_history import ridge_predictions as _positive_rank_predictions
from primary_lmp_high import _contrast
from primary_pca import prepare_split, projected_covariance as _positive_rank_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_target_covariance_svd_ridge_v1", "lmp_target_covariance_svd_basis_labels8_v1")
ANCHOR = "lmp_full_channel_ridge_v1"
REPLICATES = tuple(range(8))


def numeric_trial_ids(ids, session):
    """Canonical numeric ordering while preserving native serialized identities."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError("Unique original trial identities required")
    if np.issubdtype(ids.dtype, np.integer):
        return ids
    numbers = []
    for identity in ids.tolist():
        prefix = session + "_trial"
        if not isinstance(identity, str) or not identity.startswith(prefix):
            raise ValueError("Exact original session-qualified trial identity required")
        suffix = identity[len(prefix):]
        if not suffix.isdigit() or len(suffix) != 6 or int(suffix) < 1:
            raise ValueError("Original one-based six-digit trial suffix required")
        numbers.append(int(suffix))
    return np.asarray(numbers, dtype=int)


def covariance_basis(fitting, residual):
    x, u = np.asarray(fitting, dtype=float), np.asarray(residual, dtype=float)
    if x.ndim not in (2, 3) or x.shape[-1] < 1:
        raise ValueError("Nonempty physical fitting coordinates required")
    x, u = x.reshape(-1, x.shape[-1]), u.reshape(-1, 2)
    if len(x) != len(u) or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Finite paired fitting input and native XY residual rows required")
    cross = x.T@u/len(x)
    vectors, values, _ = np.linalg.svd(cross, full_matrices=False)
    tolerance = 64*np.finfo(np.float64).eps*max(x.shape[1], 2)*max(1., float(values[0]))
    keep = values > tolerance
    basis = vectors[:, keep]
    return basis, {"singular_values": values.tolist(), "singular_value_tolerance": tolerance,
                   "retained_components": int(np.count_nonzero(keep)), "input_basis": basis.tolist(),
                   "cross_covariance": cross.tolist(), "whitened": False,
                   "native_XY_scaled": False, "full_span": basis.shape[1] == x.shape[1]}


def trial_map(ids, labels, session, held, replicate):
    """Destination-to-donor positions in fitting-only trial order, plus exact IDs."""
    ids, labels = np.asarray(ids), np.asarray(labels)
    numeric = numeric_trial_ids(ids, session)
    allowed = sorted(set(range(8))-{held})
    if (held not in range(8) or replicate not in REPLICATES or ids.ndim != 1
            or labels.shape != ids.shape
            or len(set(ids.tolist())) != len(ids) or set(labels.tolist()) != set(allowed)):
        raise ValueError("Unique integer original IDs and exactly seven fitting directions required")
    mapping = np.empty(len(ids), dtype=int)
    metadata = []
    for direction in allowed:
        positions = sorted(np.flatnonzero(labels == direction).tolist(), key=lambda i: int(numeric[i]))
        donors = list(positions)
        literal = f"ep05-lmp-covariance-svd-basis-label-v1|20261004|{session}|{held}|{replicate}|{direction}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donors)
        mapping[positions] = donors
        metadata.append({"direction": direction, "seed_literal": literal,
                         "recipient_trial_ids": ids[positions].tolist(),
                         "donor_trial_ids": ids[donors].tolist()})
    return mapping, metadata


def project_arm(z, basis, train):
    projected = z@basis
    return projected-projected[train].mean(axis=(0, 1))


def projected_covariance(fitting):
    if fitting.shape[-1]:
        return _positive_rank_covariance(fitting)
    return {"eigenvalues": [], "trace": 0., "numerical_rank": 0,
            "numerical_rank_tolerance": 0.,
            "ridge_df_by_lambda": [{"lambda": p, "effective_df": 0.} for p in LAMBDAS],
            "diagnostic_only": True}


def ridge_predictions(fitting, evaluation, residual, check=lambda: None):
    if fitting.shape[-1] == 0:
        check()
        return np.zeros((len(LAMBDAS), int(np.prod(evaluation.shape[:-1])), 2))
    return _positive_rank_predictions(fitting, evaluation, residual, check)


def average_control_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight fixed basis-label replicates required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError(f"Basis-control support differs: {key}")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("Basis-control penalty identities differ")
        correct = float(np.mean([row["candidate_SSE"] for row in seeds]))
        wrong = float(np.mean([row["wrong_pair_SSE"] for row in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
                     "candidate_SSE": correct, "wrong_pair_SSE": wrong,
                     "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
                     "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
            "basis_label_replicate_ids": list(REPLICATES), "basis_label_Monte_Carlo_draws": 8,
            "basis_label_seed_rows": [{"replicate": r, "lambda_rows": b["lambda_rows"]}
                                      for r, b in zip(REPLICATES, banks)],
            "basis_label_aggregation": "mean correct/wrong SSE; not averaged predictions"}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or len(set(ids.tolist())) != len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or set(labels.tolist()) != set(range(8))
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
        basis, basis_info = covariance_basis(z[train], true_u)
        bases, diagnostics, maps, overlaps = [basis_info], [], [], []
        for arm in range(1+len(REPLICATES)):
            check()
            arm_basis = basis
            if arm:
                mapping, details = trial_map(ids[train], labels[train], session, held, arm-1)
                permuted = true_u.reshape(-1, 15, 2)[mapping]
                arm_basis, info = covariance_basis(z[train], permuted)
                bases.append(info)
                maps.append({"replicate": arm-1, "directions": details})
                overlaps.append(float(np.square(basis.T@arm_basis).sum()))
            projected = project_arm(z, arm_basis, train)
            diagnostics.append(projected_covariance(projected[train]))
            # The control map affects the basis only. Every head fits TRUE U.
            fitted = ridge_predictions(projected[train], projected[test], true_u, check)
            for index in range(len(LAMBDAS)):
                value = base+fitted[index].reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(true_u),
            "physical_mean": state["physical_mean"].tolist(), "physical_sd": state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "paired_basis": bases[0], "control_bases_by_replicate": bases[1:],
            "paired_projected_covariance": diagnostics[0],
            "control_projected_covariance_by_replicate": diagnostics[1:],
            "basis_trial_maps_by_replicate": maps, "projector_overlap_by_replicate": overlaps})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    all_bases = [b for row in metadata for b in [row["paired_basis"], *row["control_bases_by_replicate"]]]
    candidate.update(physical_electrode_count=x.shape[-1], direction_fit_metadata=metadata,
        covariance_SVD_fits=72, nominal_component_cap=2, whitened=False, native_XY_scaled=False,
        all_arms_all_folds_full_span=all(b["full_span"] for b in all_bases),
        all_arms_all_folds_zero_rank=all(b["retained_components"] == 0 for b in all_bases))
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    control = average_control_banks(controls)
    control.update(basis_refitted_after_permutation=True, physical_normalization_preserved=True,
                   ridge_heads_use_true_targets=True, realized_ranks_independent=True,
                   diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != len(SESSIONS)
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete sixteen-session candidate/control/reference banks required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete covariance-SVD/basis-control recipes required")
        for recipe, bank in [*banks.items(), (ANCHOR, anchor_banks[session])]:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original lambda scores required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("Covariance-SVD/control/reference baseline or target support differs")
        if banks[RECIPES[1]]["basis_label_replicate_ids"] != list(REPLICATES):
            raise ValueError("All eight fixed basis-label replicates required")
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
                seeds = bank["basis_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed-level scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            if recipe == RECIPES[0] and "direction_fit_metadata" in bank:
                def diagnostic(basis, covariance):
                    return {"retained_components": basis["retained_components"],
                            "singular_values": basis["singular_values"],
                            "singular_value_tolerance": basis["singular_value_tolerance"],
                            "trace": covariance["trace"], "numerical_rank": covariance["numerical_rank"],
                            "effective_df": covariance["ridge_df_by_lambda"][chosen]["effective_df"]}
                row["selected_projection_diagnostics"] = [{"held_direction": d["held_direction"],
                    "paired": diagnostic(d["paired_basis"], d["paired_projected_covariance"]),
                    "controls": [diagnostic(b, c) for b, c in zip(d["control_bases_by_replicate"], d["control_projected_covariance_by_replicate"])],
                    "projector_overlap_by_replicate": d["projector_overlap_by_replicate"]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        anchor_matched.append({"session_id": held, "animal": ANIMALS[held], **anchor_banks[held]["lambda_rows"][chosen]})
    full_span = all(paired_banks[s]["all_arms_all_folds_full_span"] for s in SESSIONS)
    zero_rank = all(paired_banks[s]["all_arms_all_folds_zero_rank"] for s in SESSIONS)
    duplicate = full_span or zero_rank
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      scientific_trial_count_increment=0 if duplicate else 1, promotion_authorized=False,
                      search_complete=False, evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "Covariance-SVD minus mean basis-label control")
    contrasts = {"independently_selected": _contrast(selected[RECIPES[0]], anchor_panel["sessions"], "Covariance-SVD minus independently selected LMP"),
                 "covariance_svd_lambda_matched": _contrast(selected[RECIPES[0]], anchor_matched, "Covariance-SVD minus LMP at candidate-selected lambda")}
    screens = {k: c["all_directional_screens_pass"] for k, c in contrasts.items()}
    screens["basis_labels"] = control["all_directional_screens_pass"]
    return {"panels": panels, "basis_label_control": control, "anchor_contrasts": contrasts,
            "matched_lambda_anchor_sessions": anchor_matched, "compression_screens": screens,
            "all_compression_screens_pass": all(screens.values()), "structural_duplicate": duplicate,
            "structural_full_span_duplicate": full_span, "structural_baseline_duplicate": zero_rank,
            "scientific_trial_count_increment": 0 if duplicate else 2}
