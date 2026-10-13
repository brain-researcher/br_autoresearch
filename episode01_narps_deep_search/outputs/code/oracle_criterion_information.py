#!/usr/bin/env python3
"""EP01 actual-offer oracle information for the fixed-bound MC SESOI.

This diagnostic answers a recovery-design question, not a fitted empirical
comparison. It uses all original offers and the previously drawn full-cohort
nuisance parameters. It calibrates c separately at each participant's zero-
drift reference and actual four-second decision horizon, then integrates the
joint choice/RT/deadline likelihood with the recorded contaminant mixture.

The null comparator has the same nuisance parameters and c=0. The best null
can only be closer in population KL. This is not a bound on the noisy finite-
sample score of a fitted null, and does not determine a recovery probability.
Both DDM-only and mixture-level interpretations of the 5pp calibration are
reported; no favorable interpretation is selected after the calculation.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

import deadline_ddm as ddm
from qualify_assets import require_compute


def calibrate_criterion(a, t0, fraction, channel):
    decision_deadline = 4-t0
    multiplier = 1. if channel == "ddm" else 1-fraction
    reference = float(ddm.lower_cdf(decision_deadline, 0., a, .5))
    target = reference-.05/multiplier
    if not 0 < target < reference:
        raise ValueError("Reference response probability cannot support a five-point decrease")
    def residual(c):
        return float(ddm.lower_cdf(decision_deadline, -c, a, .5))-target
    bracket = -1/a
    for _ in range(20):
        if residual(bracket) < 0:
            return brentq(residual, bracket, 0., xtol=1e-12)
        bracket *= 2
    raise RuntimeError("Could not bracket the registered criterion calibration")


def integrate_information(v0, c, a, t0, fraction, contaminant_miss, order):
    horizon = 4-t0
    nodes, weights = np.polynomial.legendre.leggauss(order)
    decision_times = horizon*(nodes+1)/2
    weights = horizon*weights/2
    v0 = np.asarray(v0)[:, None]
    v1 = v0+c
    times = decision_times[None, :]
    uniform_density = fraction*(1-contaminant_miss)/8
    kl = np.zeros(len(v0))
    true_response_mass = np.full(len(v0), 2*uniform_density*t0)
    null_response_mass = np.full(len(v0), 2*uniform_density*t0)
    for choice in [0, 1]:
        p = (1-fraction)*ddm.response_density(times, v1, a, .5, choice)+uniform_density
        q = (1-fraction)*ddm.response_density(times, v0, a, .5, choice)+uniform_density
        kl += np.sum(weights*p*np.log(p/q), axis=1)
        true_response_mass += p@weights
        null_response_mass += q@weights
    true_survival = (1-fraction)*ddm.survival(horizon, v1[:, 0], a, .5)+fraction*contaminant_miss
    null_survival = (1-fraction)*ddm.survival(horizon, v0[:, 0], a, .5)+fraction*contaminant_miss
    kl += true_survival*np.log(true_survival/null_survival)
    mean_stopping_time = ddm.survival(times, v1, a, .5)@weights
    path_bound = (1-fraction)*.5*c*c*mean_stopping_time
    universal_bound = (1-fraction)*.5*c*c*min(horizon, a*a/4)
    mass_error = max(float(np.max(np.abs(true_response_mass+true_survival-1))),
                     float(np.max(np.abs(null_response_mass+null_survival-1))))
    return {"observed_joint_kl": kl, "full_path_bound": path_bound,
            "universal_bound": universal_bound, "probability_mass_error": mass_error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--generating-parameters", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Use a new diagnostic attempt directory")
    truth = json.loads(args.generating_parameters.read_text())
    spec = json.loads(args.implementation.read_text())
    by_participant = defaultdict(list)
    with gzip.open(args.events, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            by_participant[row["participant"]].append({"gain": float(row["gain"]),
                "loss": float(row["loss"]), "version": row["task_version"]})
    if len(by_participant) != 108 or set(by_participant) != set(truth):
        raise ValueError("The diagnostic requires the complete existing 108-participant nuisance draw")
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MC_SESOI_oracle_information",
              "job_id": os.environ["SLURM_JOB_ID"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_event_table": str(args.events), "nuisance_draw": str(args.generating_parameters),
              "contaminant": spec["contaminant"], "subjects": 108, "offers_per_subject": 256,
              "generating_family": "MC", "z": .5, "eta": 0., "SESOI_level": 1.,
              "declared_selection_margin_nats_per_trial": .01,
              "fitted_models": False, "empirical_response_values_used": False,
              "recovery_probability_estimated": False, "behavioral_gate_verdict": "not_evaluated",
              "scope": "one_recorded_nuisance_draw_fixed_bounds_MC_only_not_all_EP01_generators",
              "interpretation_limit": "Oracle expected joint-likelihood separation from same-nuisance M0; best population M0 can be closer. Finite-sample fitted score and 2SE/90percent recovery are not established.",
              "participants": []}
    try:
        eps = spec["contaminant"]["fraction"]
        miss = spec["contaminant"]["miss_probability"]
        for participant in sorted(by_participant):
            rows = by_participant[participant]
            if len(rows) != 256:
                raise ValueError("Participant's original offer matrix is incomplete")
            parameters = truth[participant]
            a = float(np.exp(parameters["log_a"]))
            t0 = float(4*expit(parameters["nondecision_logit"]))
            v0 = ddm.drift_rate([r["gain"] for r in rows], [r["loss"] for r in rows], parameters["kappa"])
            for channel in ["ddm", "observed_mixture"]:
                c = calibrate_criterion(a, t0, eps, channel)
                previous, result, converged, difference = None, None, False, None
                for order in [64, 128, 256, 512]:
                    result = integrate_information(v0, c, a, t0, eps, miss, order)
                    if previous is not None:
                        difference = float(np.max(np.abs(result["observed_joint_kl"]-previous)))
                        if difference <= 1e-7 and result["probability_mass_error"] <= 1e-7:
                            converged = True
                            break
                    previous = result["observed_joint_kl"]
                row = {"participant": participant, "version": rows[0]["version"],
                       "calibration_channel": channel, "a": a, "t0": t0, "criterion": c,
                       "criterion_times_boundary": c*a,
                       "mean_observed_joint_KL_nats_per_trial": float(result["observed_joint_kl"].mean()),
                       "mean_full_path_KL_bound": float(result["full_path_bound"].mean()),
                       "universal_KL_bound": float(result["universal_bound"]),
                       "quadrature_order": order, "quadrature_converged": converged,
                       "maximum_successive_KL_change": difference,
                       "probability_mass_error": result["probability_mass_error"]}
                if result["observed_joint_kl"].min() < -1e-7:
                    raise FloatingPointError("Negative KL outside numerical tolerance")
                if np.max(result["observed_joint_kl"]-result["full_path_bound"]) > 1e-6:
                    raise FloatingPointError("Observed KL exceeds the full-path information bound")
                report["participants"].append(row)
        report["summary"] = []
        for channel in ["ddm", "observed_mixture"]:
            for version in ["EI", "ER", "both"]:
                rows = [r for r in report["participants"] if r["calibration_channel"] == channel
                        and (version == "both" or r["version"] == version)]
                report["summary"].append({"calibration_channel": channel, "version": version,
                    "participants": len(rows), "all_quadratures_converged": all(r["quadrature_converged"] for r in rows),
                    "equal_participant_mean_oracle_KL": float(np.mean([r["mean_observed_joint_KL_nats_per_trial"] for r in rows])),
                    "equal_participant_mean_path_bound": float(np.mean([r["mean_full_path_KL_bound"] for r in rows])),
                    "equal_participant_mean_universal_bound": float(np.mean([r["universal_KL_bound"] for r in rows]))})
        report["status"] = "oracle_information_computed" if all(r["quadrature_converged"] for r in report["participants"]) else "quadrature_unresolved"
    except Exception as exc:
        report.update(status="diagnostic_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir/"oracle_information.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({key: report[key] for key in ["status", "summary", "error", "elapsed_seconds"] if key in report}, indent=2))


if __name__ == "__main__":
    main()
