"""Fixed observed-behavior bridge and actual-design diagnostics for EP01.

Run numerical functions in the coordinator's verified Slurm allocation. This
module has no DDM, BOLD reader, model-selection step, or automatic rank repair.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from nilearn.glm.first_level.design_matrix import create_cosine_drift
from nilearn.glm.first_level.hemodynamic_models import compute_regressor

from run_observed_behavior import HALVES, split_people


ARMS = ("S00", "S10", "S01", "S11", "R11")
RESPONSE_CATEGORIES = ("strong_accept", "weak_accept", "weak_reject", "strong_reject")
PENALTY = np.array([0.01, 1.0, 1.0])
HIGH_PASS_HZ = 1.0 / 128.0


def _ridge_logistic(x, y, penalty, check):
    if len(y) == 0:
        raise ValueError("No responded training trials; choice prediction is unavailable")

    def objective(beta):
        check()
        linear = x @ beta
        loss = np.sum(np.logaddexp(0.0, linear) - y * linear)
        loss += 0.5 * np.dot(penalty * beta, beta)
        gradient = x.T @ (expit(linear) - y) + penalty * beta
        return float(loss), gradient

    fitted = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"gtol": 1e-9, "ftol": 1e-14, "maxiter": 1000, "maxls": 50})
    # Relative objective changes can reach machine precision before the fixed
    # gradient check passes. Polish the same strictly convex objective using
    # its exact Hessian; every coefficient retains its original positive prior.
    beta = fitted.x.copy()
    loss, gradient = objective(beta)
    polish_steps = 0
    for _ in range(25):
        check()
        if np.max(np.abs(gradient)) <= 1e-8:
            break
        probabilities = expit(x @ beta)
        weight = probabilities * (1.0-probabilities)
        hessian = x.T @ (weight[:, None]*x) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        decrement = float(gradient @ step)
        if not np.isfinite(decrement) or decrement <= 0:
            raise RuntimeError("Fixed logistic Newton direction is not finite descent")
        accepted = False
        for backtrack in range(30):
            scale = 0.5**backtrack
            candidate = beta-scale*step
            candidate_loss, candidate_gradient = objective(candidate)
            roundoff = 8*np.finfo(float).eps*max(1.0, abs(loss))
            armijo = candidate_loss <= loss-1e-4*scale*decrement
            rounded_descent = (candidate_loss <= loss+roundoff
                               and np.max(np.abs(candidate_gradient)) < np.max(np.abs(gradient)))
            if np.isfinite(candidate_loss) and (armijo or rounded_descent):
                beta, loss, gradient = candidate, candidate_loss, candidate_gradient
                polish_steps += 1
                accepted = True
                break
        if not accepted:
            break
    gradient_max = float(np.max(np.abs(gradient)))
    if not np.all(np.isfinite(beta)) or not np.isfinite(gradient_max) or gradient_max > 1e-6:
        raise RuntimeError(f"Fixed logistic fit did not converge: {fitted.message}; gradient={gradient_max}")
    return beta, {"iterations": int(fitted.nit), "gradient_max": gradient_max,
                      "lbfgs_success": bool(fitted.success), "newton_polish_steps": polish_steps,
                      "newton_polish_target": 1e-8, "accepted_gradient_limit": 1e-6,
                      "training_responded": int(len(y)), "training_classes": sorted(set(y.astype(int).tolist()))}


def _features(rows):
    return np.asarray([[1.0, float(row["gain"]) / 10.0, float(row["loss"]) / 10.0]
                       for row in rows], dtype=float)


def behavioral_covariates(people, check=lambda: None):
    """Return (copied people with cross-fitted row covariates, metadata).

    Preserve the reader's {participant: {version, rows}} representation. Choice
    likelihood uses responded trials only. Every original row, including each
    deadline miss, receives predictions from the opposite run half.
    """
    augmented = {person: {"version": panel["version"], "rows": [dict(row) for row in panel["rows"]]}
                 for person, panel in people.items()}
    lookup = {person: {(int(row["run"]), int(row["trial_order"])): row for row in panel["rows"]}
              for person, panel in augmented.items()}
    participants = {person: {"task_version": panel["version"], "directions": []}
                    for person, panel in people.items()}
    for direction in range(2):
        training, scoring = split_people(people, direction)
        for person in sorted(people):
            check()
            train_rows = [row for row in training[person]["rows"] if int(row["choice"]) in (0, 1)]
            y = np.asarray([int(row["choice"]) for row in train_rows], dtype=float)
            beta, fit_info = _ridge_logistic(_features(train_rows), y, PENALTY, check)
            intercept, intercept_info = _ridge_logistic(np.ones((len(y), 1)), y, PENALTY[:1], check)
            score_rows = scoring[person]["rows"]
            linear = _features(score_rows) @ beta
            probabilities = expit(linear)
            entropy = -xlogy(probabilities, probabilities) - xlogy(1-probabilities, 1-probabilities)
            responded = np.asarray([int(row["choice"]) in (0, 1) for row in score_rows])
            score_y = np.asarray([int(row["choice"]) for row in score_rows], dtype=float)[responded]
            score_linear = linear[responded]
            model_nll = float(np.sum(np.logaddexp(0.0, score_linear) - score_y * score_linear))
            intercept_nll = float(np.sum(np.logaddexp(0.0, intercept[0]) - score_y * intercept[0]))
            for row, probability, uncertainty in zip(score_rows, probabilities, entropy):
                target = lookup[person][int(row["run"]), int(row["trial_order"])]
                missed = int(row["no_response"]) == 1
                duration = 4.0 if missed else float(row["response_time"])
                target.update(p_accept=float(probability), choice_entropy=float(uncertainty),
                              choice_fit_direction=direction, decision_duration=duration,
                              response_onset=None if missed else float(row["onset"])+duration,
                              response_or_deadline_onset=float(row["onset"])+duration)
            participants[person]["directions"].append({
                "direction": direction, "train_runs": list(HALVES[direction][0]),
                "score_runs": list(HALVES[direction][1]), "coefficients": beta.tolist(),
                "intercept_reference": float(intercept[0]), "fit": fit_info,
                "intercept_fit": intercept_info, "scoring_responded": int(responded.sum()),
                "model_nll": model_nll, "intercept_nll": intercept_nll})
    for person, panel in augmented.items():
        check()
        run_diagnostics = {}
        for run in range(1, 5):
            rows = [row for row in panel["rows"] if int(row["run"]) == run]
            entropy = np.asarray([row["choice_entropy"] for row in rows])
            mean, sd = float(np.mean(entropy)), float(np.std(entropy, ddof=0))
            scaled = np.zeros(len(rows)) if sd == 0.0 else (entropy-mean)/sd
            for row, value in zip(rows, scaled):
                row["entropy_z"] = float(value)
            run_diagnostics[str(run)] = {"entropy_mean": mean, "entropy_sd": sd,
                                         "entropy_zero_sd": sd == 0.0}
        record = participants[person]
        count = sum(item["scoring_responded"] for item in record["directions"])
        if not count:
            raise ValueError(f"{person}: no responded scoring trials")
        record.update(scoring_responded=count, runs=run_diagnostics,
                      model_log_loss=sum(item["model_nll"] for item in record["directions"])/count,
                      intercept_log_loss=sum(item["intercept_nll"] for item in record["directions"])/count)
        record["log_loss_improvement"] = record["intercept_log_loss"]-record["model_log_loss"]
    by_version = defaultdict(list)
    for record in participants.values():
        by_version[record["task_version"]].append(record)
    summaries = {}
    for name, records in [("all", list(participants.values())), *sorted(by_version.items())]:
        summaries[name] = {key: float(np.mean([record[key] for record in records]))
                           for key in ("model_log_loss", "intercept_log_loss", "log_loss_improvement")}
        summaries[name]["participants"] = len(records)
    return augmented, {"method": "fixed_per_participant_opposite_half_ridge_logistic",
                       "features": ["intercept", "gain/10", "positive_loss/10"],
                       "penalty_diagonal": PENALTY.tolist(),
                       "probability_interpretation": "accept_given_response_and_offer",
                       "entropy_interpretation": "predicted_choice_uncertainty_conditional_on_responding",
                       "loss_denominator": "responded_trials_within_participant_then_equal_participant_mean",
                       "participants": participants, "summary": summaries}


def design_diagnostics(design, task_columns, check=lambda: None):
    """Diagnose the untouched design; efficiencies use unit-variance OLS geometry."""
    check()
    x = design.to_numpy(dtype=float)
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    tolerance = np.finfo(float).eps * max(x.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    basis = vt[:rank]
    focal = {}
    for name in ("gain", "loss"):
        index = design.columns.get_loc(name)
        unit = np.zeros(x.shape[1]); unit[index] = 1.0
        error = float(np.linalg.norm(unit-basis.T @ (basis @ unit)))
        estimable = error <= 1e-8
        variance = float(np.sum((basis[:, index]/singular[:rank])**2)) if estimable else None
        focal[name] = {"estimable": bool(estimable), "row_space_error": error,
                       "variance": variance,
                       "efficiency": 1.0/variance if variance is not None and variance > 0 else None}
    nuisance_columns = [name for name in design if name not in task_columns]
    task = design.loc[:, task_columns].to_numpy(dtype=float)
    nuisance = design.loc[:, nuisance_columns].to_numpy(dtype=float)
    residual = task - nuisance @ np.linalg.lstsq(nuisance, task, rcond=None)[0]
    norms = np.linalg.norm(residual, axis=0)
    original_norms = np.linalg.norm(task, axis=0)
    active = norms > 1e-10*np.maximum(original_norms, 1.0)
    scaled = residual[:, active]/norms[active] if active.any() else np.empty((len(x), 0))
    residual_singular = np.linalg.svd(scaled, compute_uv=False)
    residual_rank = int(np.sum(residual_singular > np.finfo(float).eps*max(scaled.shape)*residual_singular[0])) if len(residual_singular) else 0
    finite_condition = bool(len(residual_singular) and residual_rank == int(active.sum()))
    condition = float(residual_singular[0]/residual_singular[-1]) if finite_condition else None
    absorbed = [name for name, norm, keep in zip(task_columns, original_norms, active)
                if norm > 1e-10 and not keep]
    for name in ("gain", "loss"):
        index = task_columns.index(name)
        others = np.delete(residual, index, axis=1)
        y = residual[:, index]
        remaining = y-others @ np.linalg.lstsq(others, y, rcond=None)[0]
        denominator = float(remaining @ remaining)
        vif = float((y@y)/denominator) if focal[name]["estimable"] and denominator > 0 else None
        focal[name]["vif"] = vif
    return {"rows": int(x.shape[0]), "columns": int(x.shape[1]), "rank": rank,
            "rank_tolerance": float(tolerance), "residual_degrees_of_freedom": int(len(x)-rank),
            "task_columns": list(task_columns), "nuisance_columns": nuisance_columns,
            "zero_columns": [name for index, name in enumerate(design) if np.all(x[:, index] == 0)],
            "nuisance_residualized_zero_task_columns": [name for name, keep in zip(task_columns, active) if not keep],
            "nuisance_absorbed_nonzero_task_columns": absorbed,
            "task_condition_number": condition, "task_condition_infinite": not finite_condition,
            "task_condition_pass": bool(condition is not None and condition <= 30 and not absorbed),
            "focal": focal, "focal_estimable": {name: value["estimable"] for name, value in focal.items()},
            "estimable_gain": focal["gain"]["estimable"], "estimable_loss": focal["loss"]["estimable"],
            "focal_vif_pass": all(value["vif"] is not None and value["vif"] <= 5 for value in focal.values()),
            "efficiency_definition": "inverse_unit_noise_OLS_contrast_variance_no_AR_weighting",
            "design_modified_for_rank": False}


def build_design(rows, frame_times, confounds, arm, check=lambda: None):
    """Return full named design and diagnostics without dropping zero task columns.

    confounds is the runner's fixed nuisance DataFrame (no drift/intercept).
    frame_times must be the actual NIfTI-header TR times scan indices.
    """
    check()
    if arm not in ARMS or not rows:
        raise ValueError("A supported arm and nonempty original run are required")
    times = np.asarray(frame_times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
            or not np.allclose(np.diff(times), times[1]-times[0], rtol=1e-10, atol=1e-10)):
        raise ValueError("frame_times must be finite, increasing and uniformly sampled")
    versions = {row["task_version"] for row in rows}
    if len(versions) != 1 or not versions.issubset({"EI", "ER"}):
        raise ValueError("One original task version is required per run")
    if len({(row["participant"], int(row["run"])) for row in rows}) != 1:
        raise ValueError("One original participant/run is required")
    onset = np.asarray([float(row["onset"]) for row in rows])
    missed = np.asarray([int(row["no_response"]) == 1 for row in rows])
    duration = np.asarray([4.0 if miss else float(row["response_time"]) for row, miss in zip(rows, missed)])
    gain = np.asarray([float(row["gain"]) for row in rows])-(25.0 if "EI" in versions else 12.5)
    loss = np.asarray([float(row["loss"]) for row in rows])-12.5
    zero = np.zeros(len(rows)); one = np.ones(len(rows))
    value_onset = onset+duration if arm == "R11" else onset
    conditions = [("offer", onset, zero, one), ("gain", value_onset, zero, gain),
                  ("loss", value_onset, zero, loss), ("miss", onset[missed], zero[missed], one[missed])]
    if arm in ("S10", "S11", "R11"):
        conditions.extend([("entropy", onset, zero, np.asarray([float(row["entropy_z"]) for row in rows])),
                           ("decision_epoch", onset, duration, one)])
    if arm in ("S01", "S11", "R11"):
        for category in RESPONSE_CATEGORIES:
            selected = np.asarray([row["response_category"] == category for row in rows])
            conditions.append(("response_"+category, (onset+duration)[selected], zero[selected], one[selected]))
    columns = {}
    for name, event_onsets, event_durations, amplitudes in conditions:
        check()
        if len(event_onsets) == 0 or np.all(amplitudes == 0):
            columns[name] = np.zeros(len(times))
        else:
            values, names = compute_regressor(np.vstack([event_onsets, event_durations, amplitudes]),
                                             hrf_model="spm", frame_times=times, con_id=name,
                                             oversampling=50, min_onset=-24)
            if values.shape != (len(times), 1) or names != [name]:
                raise ValueError("Expected exactly one canonical HRF column per condition")
            columns[name] = values[:, 0]
    task_columns = list(columns)
    if not isinstance(confounds, pd.DataFrame) or len(confounds) != len(times):
        raise ValueError("Confounds must be a DataFrame aligned to all original frames")
    if not confounds.columns.is_unique:
        raise ValueError("Duplicate confound names")
    for name in confounds:
        if not isinstance(name, str) or name in columns or name == "constant" or name.startswith("drift_"):
            raise ValueError(f"Confound name collides with design columns: {name}")
        columns[name] = confounds[name].to_numpy(dtype=float)
    drift = create_cosine_drift(HIGH_PASS_HZ, times)
    for index in range(drift.shape[1]-1):
        columns[f"drift_{index+1}"] = drift[:, index]
    columns["constant"] = drift[:, -1]
    design = pd.DataFrame(columns, index=times)
    if not np.all(np.isfinite(design.to_numpy())):
        raise ValueError("Nonfinite design entries")
    diagnostics = design_diagnostics(design, task_columns, check)
    diagnostics.update(arm=arm, trial_count=len(rows), missed_trials=int(missed.sum()),
                       hrf="spm", oversampling=50, high_pass_hz=HIGH_PASS_HZ,
                       conditions_orthogonalized_against_each_other=False,
                       response_category_counts={category: sum(row["response_category"] == category for row in rows)
                                                 for category in RESPONSE_CATEGORIES},
                       entropy_all_zero=all(float(row.get("entropy_z", 0)) == 0 for row in rows))
    return design, diagnostics
