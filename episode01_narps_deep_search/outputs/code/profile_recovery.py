#!/usr/bin/env python3
"""One full-design M0 recovery replicate for numerical/runtime estimation.

This is the first representative piece of the policy-required recovery work,
not a reduced-sample scientific trial and not a passing behavioral kill test.
All 108 participants, 432 runs and original offers are retained. Responses are
simulated, so this command never fits or scores observed choices/RTs.
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
import time

import numpy as np
from scipy.special import expit

import deadline_ddm
from hierarchical_behavior import fit_partially_pooled, load_split, make_data, trial_likelihood
from qualify_assets import require_compute


def truncated_normal(rng, mean, sd, bounds):
    for _ in range(10000):
        draw = rng.normal(mean, sd)
        if bounds[0] <= draw <= bounds[1]:
            return float(draw)
    raise RuntimeError("Nuisance drawing failed within declared parameter support")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=710031)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Use a new attempt path")
    spec = json.loads(args.implementation.read_text())
    if spec.get("state") != "recorded_before_model_scoring" or spec.get("policy_version") != 6:
        raise ValueError("An instantiated policy-v6 specification is required")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    report = {"analysis_id": "behavioral_mechanism_kill_test",
              "phase": "full_design_M0_recovery_runtime_replicate",
              "job_id": os.environ["SLURM_JOB_ID"], "seed": args.seed,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "observed_responses_fit_or_scored": False, "behavioral_gate_evaluable": False,
              "source_event_table": str(args.events), "implementation": spec,
              "generating_regime": "independent_truncated_normal_nuisance_draws_at_recorded_population_centers_and_individual_scales",
              "censoring": "model_generated_deadline_survival_and_identical_contaminant_channel_not_forced_observed_missingness",
              "qualification_limit": "This one M0 runtime replicate cannot establish model selection, coverage, recovery or robustness rates."}
    try:
        by_participant = defaultdict(list)
        with gzip.open(args.events, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                by_participant[row["participant"]].append(row)
        if len(by_participant) != 108 or sum(map(len, by_participant.values())) != 27648:
            raise ValueError("The representative replicate requires the complete registered design")
        rng = np.random.default_rng(args.seed)
        truth, simulated = {}, []
        for participant in sorted(by_participant):
            rows = by_participant[participant]
            if len(rows) != 256 or {int(r["run"]) for r in rows} != {1, 2, 3, 4}:
                raise ValueError("Participant source design is incomplete")
            parameters = {}
            for name in ["kappa", "log_a", "nondecision_logit"]:
                prior = spec["parameters"][name]
                parameters[name] = truncated_normal(rng, prior["population_center"],
                    prior["individual_scale_center"], prior["bounds"])
            truth[participant] = parameters
            drift = deadline_ddm.drift_rate([float(r["gain"]) for r in rows],
                [float(r["loss"]) for r in rows], parameters["kappa"])
            choices, rt = deadline_ddm.simulate(rng, drift, np.exp(parameters["log_a"]), .5,
                4*expit(parameters["nondecision_logit"]), spec["contaminant"]["fraction"],
                spec["contaminant"]["miss_probability"])
            for row, choice, response_time in zip(rows, choices, rt):
                record = dict(row)
                record.update(choice=int(choice), no_response=int(choice == -1), strength=-1,
                    response_time="" if choice == -1 else float(response_time),
                    response_category={-1: "no_response", 0: "simulated_reject", 1: "simulated_accept"}[int(choice)])
                simulated.append(record)
        simulation_seconds = time.perf_counter()-started
        synthetic_path = args.output_dir/"simulated_events.csv.gz"
        with gzip.open(synthetic_path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(simulated[0]))
            writer.writeheader()
            writer.writerows(simulated)
        (args.output_dir/"generating_parameters.json").write_text(json.dumps(truth, indent=2)+"\n")
        report.update(participants=108, runs=432, trials=27648,
                      simulated_missing_responses=sum(row["no_response"] for row in simulated),
                      simulation_seconds=simulation_seconds, fits=[])
        for training_runs, scoring_runs in [([1, 4], [2, 3]), ([2, 3], [1, 4])]:
            training, scoring = load_split(synthetic_path, training_runs, scoring_runs)
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, "M0", "fixed", spec)
            fit["training_runs"] = training_runs
            fit["scoring_runs"] = scoring_runs
            fit["elapsed_seconds"] = time.perf_counter()-fit_started
            if fit["converged"]:
                # Aggregate synthetic score is only a numerical check of the
                # full predictive path, never a mechanism comparison.
                logp = [float(trial_likelihood(theta, fit["parameter_names"], make_data(scoring[p]["rows"]),
                                               spec, "fixed").sum())
                        for p, theta in zip(fit["participants"], fit["parameters"])]
                fit["synthetic_total_log_predictive_density"] = sum(logp)
            report["fits"].append(fit)
            (args.output_dir/"recovery_profile.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({"training_runs": training_runs, "converged": fit["converged"],
                              "elapsed_seconds": fit["elapsed_seconds"],
                              "sweeps": len(fit["objective_history"])}), flush=True)
            if not fit["converged"]:
                raise RuntimeError("Representative hierarchical recovery fit did not converge")
        report["status"] = "representative_recovery_complete_full_gate_not_evaluated"
    except Exception as exc:
        report.update(status="representative_recovery_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir/"recovery_profile.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k: report[k] for k in ["status", "elapsed_seconds"] if k in report}), flush=True)


if __name__ == "__main__":
    main()
