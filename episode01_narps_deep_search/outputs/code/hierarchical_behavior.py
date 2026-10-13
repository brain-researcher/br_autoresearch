"""Training-only partially pooled EP01 model fitting.

Each task version has its own parameter mean and scale. Version means shrink
toward a shared mean; individual transformed parameters shrink only toward
their version's mean. All hyperparameters are learned from training runs.

This is penalized joint-mode inference, not posterior sampling. Its plug-in
predictive density is a complete probability distribution. Recovery must
qualify group intervals and individual ranks before any mechanistic or neural
claim; this module does not manufacture those qualifications.

Priors/support, contaminant distribution and numerical tolerances are supplied
in an explicit implementation JSON. No undocumented scientific defaults are
selected by this module. The exponential-bound numerical rival is included,
but its solver resolution still needs parameter-range convergence evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

import collapsing_ddm
import deadline_ddm
from qualify_assets import require_compute


FOCAL = {"M0": [], "MV": ["eta"], "MZ": ["rho"], "MC": ["criterion"],
         "MVZ": ["eta", "rho"], "MVC": ["eta", "criterion"],
         "MZC": ["rho", "criterion"], "MVZC": ["eta", "rho", "criterion"]}


def parameter_names(model, rival):
    names = ["kappa", "log_a", "nondecision_logit"] + FOCAL[model]
    if rival == "nonlinear":
        names += ["gamma_gain", "gamma_loss"]
    elif rival == "urgency":
        names += ["urgency"]
    elif rival != "fixed":
        raise ValueError("Rival must be fixed, nonlinear or urgency")
    return names


def trial_likelihood(theta, names, data, spec, rival):
    params = dict(zip(names, theta))
    a = np.exp(params["log_a"])
    z = expit(params.get("rho", 0.))
    t0 = 4*expit(params["nondecision_logit"])
    drift = deadline_ddm.drift_rate(data["gain"], data["loss"], params["kappa"],
        params.get("eta", 0.), params.get("criterion", 0.),
        params.get("gamma_gain", 1.), params.get("gamma_loss", 1.))
    eps = spec["contaminant"]["fraction"]
    miss_mass = spec["contaminant"]["miss_probability"]
    if rival != "urgency":
        return deadline_ddm.trial_log_likelihood(data["rt"], data["choice"], drift,
                                                a, z, t0, eps, miss_mass)
    logp = np.empty_like(drift)
    for value in np.unique(drift):
        rows = drift == value
        dist = collapsing_ddm.solve(float(value), a, z, params["urgency"], 4-t0,
            dt=spec["urgency_solver"]["dt_seconds"],
            intervals=spec["urgency_solver"]["spatial_intervals"])
        rt, choice = data["rt"][rows], data["choice"][rows]
        probabilities = np.empty(len(rt))
        misses = choice == -1
        probabilities[misses] = (1-eps)*dist.survival_mass+eps*miss_mass
        probabilities[~misses] = ((1-eps)*dist.response_density(rt[~misses]-t0, choice[~misses])
                                  +eps*(1-miss_mass)/8)
        with np.errstate(divide="ignore"):
            logp[rows] = np.log(probabilities)
    return logp


def make_data(rows):
    return {"gain": np.array([float(r["gain"]) for r in rows]),
            "loss": np.array([float(r["loss"]) for r in rows]),
            "rt": np.array([float(r["response_time"]) if r["response_time"] else np.nan for r in rows]),
            "choice": np.array([int(r["choice"]) for r in rows])}


def fit_partially_pooled(training, model, rival, spec):
    """Block-coordinate joint posterior mode; never receives scoring rows."""
    if model in {"MZC", "MVZC"} and not spec.get("joint_z_c_recovery_passed", False):
        raise ValueError("Observed joint z/c models require their recovery result first")
    names = parameter_names(model, rival)
    priors = [spec["parameters"][name] for name in names]
    centers = np.array([p["population_center"] for p in priors])
    pop_sd = np.array([p["population_scale"] for p in priors])
    version_sd = np.array([p["version_mean_scale"] for p in priors])
    initial_sd = np.array([p["individual_scale_center"] for p in priors])
    logsd_sd = np.array([p["individual_log_scale_sd"] for p in priors])
    bounds = [p["bounds"] for p in priors]
    if np.any(pop_sd <= 0) or np.any(version_sd <= 0) or np.any(initial_sd <= 0) or np.any(logsd_sd <= 0):
        raise ValueError("All declared prior scales must be positive")
    participants = sorted(training)
    version = np.array([0 if training[p]["version"] == "EI" else 1 for p in participants])
    if set(version) != {0, 1}:
        raise ValueError("Both task versions must be present")
    arrays = [make_data(training[p]["rows"]) for p in participants]
    theta = np.tile(centers, (len(participants), 1))
    global_mean = centers.copy()
    version_mean = np.tile(centers, (2, 1))
    logsd = np.tile(np.log(initial_sd), (2, 1))
    optimizer = spec["optimizer"]
    objectives, failures = [], []
    converged = False
    # All starts are determined by the implementation specification, never by
    # a held-out result. The same starts apply in simulated and observed fits.
    for sweep in range(optimizer["maximum_sweeps"]):
        failures = []
        for index, (participant, data) in enumerate(zip(participants, arrays)):
            group = version[index]
            mean, sd = version_mean[group], np.exp(logsd[group])
            def objective(candidate):
                try:
                    nll = -trial_likelihood(candidate, names, data, spec, rival).sum()
                except (FloatingPointError, ValueError):
                    return np.inf
                return float(nll + .5*np.sum(((candidate-mean)/sd)**2))
            starts = [theta[index]]
            if sweep == 0:
                for offset in optimizer["initial_standardized_offsets"]:
                    starts.append(np.clip(centers + np.asarray(offset[:len(names)])*initial_sd,
                                          [b[0] for b in bounds], [b[1] for b in bounds]))
            attempts = [minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": optimizer["subject_maximum_iterations"],
                         "ftol": optimizer["subject_ftol"], "gtol": optimizer["subject_gtol"]})
                        for start in starts]
            finite = [attempt for attempt in attempts if np.isfinite(attempt.fun)]
            if not finite:
                raise RuntimeError(f"No finite fit for {participant}")
            best = min(finite, key=lambda x: x.fun)
            theta[index] = best.x
            if not best.success:
                failures.append({"participant": participant, "message": str(best.message)})
        for group in [0, 1]:
            values = theta[version == group]
            variance = np.exp(2*logsd[group])
            version_mean[group] = ((values.sum(axis=0)/variance + global_mean/version_sd**2)
                                   /(len(values)/variance+1/version_sd**2))
        global_mean = ((version_mean.sum(axis=0)/version_sd**2+centers/pop_sd**2)
                       /(2/version_sd**2+1/pop_sd**2))
        for group in [0, 1]:
            values = theta[version == group]
            sumsquares = ((values-version_mean[group])**2).sum(axis=0)
            for j, prior in enumerate(priors):
                def scale_objective(log_scale):
                    return (len(values)*log_scale+.5*sumsquares[j]*np.exp(-2*log_scale)
                            +.5*((log_scale-np.log(initial_sd[j]))/logsd_sd[j])**2)
                result = minimize_scalar(scale_objective, method="bounded",
                    bounds=prior["individual_log_scale_bounds"],
                    options={"xatol": optimizer["scale_xatol"]})
                if not result.success:
                    raise RuntimeError("Hyperparameter scale fit failed")
                logsd[group, j] = result.x
        objective_total = 0.
        for i, data in enumerate(arrays):
            g = version[i]
            objective_total += -trial_likelihood(theta[i], names, data, spec, rival).sum()
            objective_total += np.sum(logsd[g]+.5*((theta[i]-version_mean[g])/np.exp(logsd[g]))**2)
        objective_total += .5*np.sum(((global_mean-centers)/pop_sd)**2)
        objective_total += .5*np.sum(((version_mean-global_mean)/version_sd)**2)
        objective_total += .5*np.sum(((logsd-np.log(initial_sd))/logsd_sd)**2)
        objectives.append(float(objective_total))
        if len(objectives) > 1:
            delta = abs(objectives[-1]-objectives[-2])
            if delta <= optimizer["joint_relative_tolerance"]*(1+abs(objectives[-2])) and not failures:
                converged = True
                break
    support_contacts = []
    for index, participant in enumerate(participants):
        for j, (name, (low, high)) in enumerate(zip(names, bounds)):
            tolerance = 1e-6*(1+high-low)
            value = float(theta[index, j])
            if value-low <= tolerance or high-value <= tolerance:
                support_contacts.append({"level": "participant", "participant": participant,
                    "parameter": name, "value": value, "bounds": [low, high]})
    for group, label in enumerate(["EI", "ER"]):
        for j, prior in enumerate(priors):
            low, high = prior["individual_log_scale_bounds"]
            tolerance = 1e-6*(1+high-low)
            value = float(logsd[group, j])
            if value-low <= tolerance or high-value <= tolerance:
                support_contacts.append({"level": "version_log_standard_deviation",
                    "version": label, "parameter": names[j], "value": value,
                    "bounds": [low, high]})
    return {"model": model, "rival": rival, "parameter_names": names,
            "inference": "partially_pooled_joint_mode_plugin_predictive",
            "participants": participants, "parameters": theta.tolist(),
            "global_mean": global_mean.tolist(), "version_means": version_mean.tolist(),
            "version_standard_deviations": np.exp(logsd).tolist(),
            "version_order": ["EI", "ER"], "objective_history": objectives,
            "converged": converged, "subject_optimizer_failures": failures,
            "support_contacts": support_contacts,
            "group_interval_coverage_qualified": False,
            "individual_rank_recovery_qualified": False}


def load_split(path, training_runs, scoring_runs):
    if set(training_runs) & set(scoring_runs):
        raise ValueError("Training and scoring runs overlap")
    if not set(training_runs+scoring_runs).issubset({1, 2, 3, 4}):
        raise ValueError("Only registered runs 1 through 4 exist")
    training, scoring = {}, {}
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            run = int(row["run"])
            destination = training if run in training_runs else scoring if run in scoring_runs else None
            if destination is not None:
                group = destination.setdefault(row["participant"], {"version": row["task_version"], "rows": []})
                if group["version"] != row["task_version"]:
                    raise ValueError("Participant changed task version")
                group["rows"].append(row)
    if set(training) != set(scoring):
        raise ValueError("Participant support differs between fit and score")
    for participant in training:
        if len(training[participant]["rows"]) != 64*len(training_runs) or len(scoring[participant]["rows"]) != 64*len(scoring_runs):
            raise ValueError("Participant/run support is incomplete")
    return training, scoring


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--model", choices=FOCAL, required=True)
    parser.add_argument("--rival", choices=["fixed", "nonlinear", "urgency"], required=True)
    parser.add_argument("--training-runs", nargs="+", type=int, required=True)
    parser.add_argument("--scoring-runs", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior attempts; use a new output directory")
    spec = json.loads(args.implementation.read_text())
    if spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring":
        raise ValueError("Record the v6 inference/recovery instantiation before model scoring")
    training, scoring = load_split(args.events, args.training_runs, args.scoring_runs)
    if len(training) != 108:
        raise ValueError("An empirical model fit requires all 108 registered participants")
    args.output_dir.mkdir(parents=True)
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "one_model_one_split",
              "job_id": os.environ["SLURM_JOB_ID"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "training_runs": args.training_runs, "scoring_runs": args.scoring_runs,
              "implementation": spec, "complete_behavioral_gate": False}
    try:
        fit = fit_partially_pooled(training, args.model, args.rival, spec)
        result["fit"] = fit
        if not fit["converged"]:
            raise RuntimeError("Hierarchical optimization did not converge; scores not released")
        scores = []
        for participant, theta in zip(fit["participants"], fit["parameters"]):
            record = scoring[participant]
            logp = trial_likelihood(theta, fit["parameter_names"], make_data(record["rows"]), spec, args.rival)
            for row, value in zip(record["rows"], logp):
                scores.append({"participant": participant, "task_version": record["version"],
                               "run": row["run"], "trial_order": row["trial_order"],
                               "sequence_template": row["sequence_template"], "log_predictive_density": float(value)})
        with gzip.open(args.output_dir/"trial_predictions.csv.gz", "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
        result.update(status="fixed_model_split_scored", scored_trials=len(scores),
                      remaining="all_other_models_splits_recovery_controls_and_group_uncertainty_required")
    except Exception as exc:
        result.update(status="fit_failed_no_gate_verdict", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir/"fit_report.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
