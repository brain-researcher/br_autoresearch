#!/usr/bin/env python3
"""One complete-design, fitted MC-at-SESOI versus M0 recovery component.

This is a necessary primary comparison in the full recovery experiment, not
the entire family/rival/sensitivity gate. Each replicate retains every source
offer and uses both primary half directions with independent training-only
hyperparameter fits. Only fully converged replicates produce a pass/fail screen.
The calibrated effect changes mixture-level acceptance by five percentage
points; this is the larger of the two calibrations checked by the oracle.
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
from oracle_criterion_information import calibrate_criterion
from profile_recovery import truncated_normal
from qualify_assets import require_compute


def summarize_paired_scores(participant_scores, seed):
    participants = sorted(participant_scores)
    effects = np.array([participant_scores[p]["MC"]-participant_scores[p]["M0"] for p in participants])
    versions = np.array([participant_scores[p]["version"] for p in participants])
    if len(participants) != 108 or {v: int(np.sum(versions == v)) for v in ["EI", "ER"]} != {"EI": 54, "ER": 54}:
        raise ValueError("Recovery comparison must retain 108 biological units and both strata")
    mean = float(effects.mean())
    se = float(effects.std(ddof=1)/np.sqrt(len(effects)))
    rng = np.random.default_rng(seed)
    resampled = np.concatenate([rng.choice(effects[versions == v], size=(9999, 54), replace=True)
                               for v in ["EI", "ER"]], axis=1)
    lower, upper = np.quantile(np.median(resampled, axis=1), [.025, .975])
    version_means = {v: float(effects[versions == v].mean()) for v in ["EI", "ER"]}
    requirements = {"mean_gain_at_least_0p01": mean >= .01,
                    "mean_gain_greater_than_two_participant_SE": mean > 2*se,
                    "positive_in_both_task_versions": all(x > 0 for x in version_means.values()),
                    "positive_participant_median": float(np.median(effects)) > 0,
                    "participant_median_95CI_lower_positive": float(lower) > 0}
    return {"mean_gain_nats_per_trial": mean, "participant_SE": se,
            "version_means": version_means, "participant_median": float(np.median(effects)),
            "participant_median_95CI": [float(lower), float(upper)],
            "bootstrap": "9999_fixed_seed_participant_draws_within_EI_ER",
            "requirements": requirements, "necessary_primary_comparison_pass": all(requirements.values()),
            "full_recovery_pass": None,
            "remaining_if_primary_passes": "other_families_urgency_nonlinear_LOO_splits_group_coverage_individual_rank_and_registered_controls"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    spec = json.loads(args.implementation.read_text())
    count = spec["recovery_plan"]["replicates_per_qualified_generating_cell"]
    if not 0 <= args.replicate_index < count:
        raise ValueError("Replicate index is outside the recorded recovery-cell count")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh replicate attempt directory")
    seed = 710031 + 10007*args.replicate_index
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MC_SESOI_fitted_recovery_component",
              "job_id": os.environ["SLURM_JOB_ID"], "replicate_index": args.replicate_index,
              "planned_cell_replicates": count, "seed": seed, "implementation": spec,
              "started_utc": datetime.now(timezone.utc).isoformat(), "fits": [],
              "generating_model": "MC", "SESOI_level": 1.,
              "calibration": "five_point_decrease_in_four_second_mixture_level_acceptance_at_zero_reference_drift",
              "nuisance_generation": "recorded_center_and_individual_scale_truncated_normal_as_profile01",
              "deadline_missingness": "generated_from_the_frozen_likelihood_all_original_trial_slots_retained",
              "observed_choices_or_RTs_used": False, "full_behavioral_gate": "not_evaluated"}
    try:
        source = defaultdict(list)
        with gzip.open(args.events, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                source[row["participant"]].append(row)
        if len(source) != 108 or sum(map(len, source.values())) != 27648:
            raise ValueError("The full registered source design is required")
        rng = np.random.default_rng(seed)
        truth, simulated = {}, []
        for participant in sorted(source):
            rows = source[participant]
            params = {}
            for name in ["kappa", "log_a", "nondecision_logit"]:
                prior = spec["parameters"][name]
                params[name] = truncated_normal(rng, prior["population_center"],
                    prior["individual_scale_center"], prior["bounds"])
            a, t0 = np.exp(params["log_a"]), 4*expit(params["nondecision_logit"])
            c = calibrate_criterion(a, t0, spec["contaminant"]["fraction"], "observed_mixture")
            params["criterion"] = c
            truth[participant] = params
            drift = deadline_ddm.drift_rate([float(r["gain"]) for r in rows],
                [float(r["loss"]) for r in rows], params["kappa"], criterion=c)
            choices, rt = deadline_ddm.simulate(rng, drift, a, .5, t0,
                spec["contaminant"]["fraction"], spec["contaminant"]["miss_probability"])
            for row, choice, response_time in zip(rows, choices, rt):
                out = dict(row)
                out.update(choice=int(choice), no_response=int(choice == -1), strength=-1,
                    response_time="" if choice == -1 else float(response_time),
                    response_category={-1: "no_response", 0: "simulated_reject", 1: "simulated_accept"}[int(choice)])
                simulated.append(out)
        synthetic_path = args.output_dir/"simulated_events.csv.gz"
        with gzip.open(synthetic_path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(simulated[0]))
            writer.writeheader()
            writer.writerows(simulated)
        (args.output_dir/"generating_parameters.json").write_text(json.dumps(truth, indent=2)+"\n")
        report["simulation_seconds"] = time.perf_counter()-started
        report["simulated_deadline_misses"] = sum(r["no_response"] for r in simulated)
        participant_scores = {p: {"M0": 0., "MC": 0., "version": rows[0]["task_version"]}
                              for p, rows in source.items()}
        for training_runs, scoring_runs in [([1, 4], [2, 3]), ([2, 3], [1, 4])]:
            training, scoring = load_split(synthetic_path, training_runs, scoring_runs)
            for model in ["M0", "MC"]:
                fit_started = time.perf_counter()
                fit = fit_partially_pooled(training, model, "fixed", spec)
                fit.update(training_runs=training_runs, scoring_runs=scoring_runs,
                           elapsed_seconds=time.perf_counter()-fit_started)
                report["fits"].append(fit)
                (args.output_dir/"criterion_recovery.json").write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps({"model": model, "training_runs": training_runs,
                                  "converged": fit["converged"], "elapsed_seconds": fit["elapsed_seconds"],
                                  "support_contacts": len(fit["support_contacts"])}), flush=True)
                if not fit["converged"]:
                    raise RuntimeError("Nonconverged fit; replicate does not count as a scientific failure")
                for participant, theta in zip(fit["participants"], fit["parameters"]):
                    logp = trial_likelihood(theta, fit["parameter_names"], make_data(scoring[participant]["rows"]), spec, "fixed")
                    participant_scores[participant][model] += float(logp.sum())/256
        report["participant_scores"] = participant_scores
        report["comparison"] = summarize_paired_scores(participant_scores, seed+4099)
        report["valid_recovery_component"] = True
        report["status"] = "necessary_primary_comparison_completed"
    except Exception as exc:
        report.update(status="technical_recovery_failure", valid_recovery_component=False,
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir/"criterion_recovery.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({key: report[key] for key in ["status", "comparison", "error", "elapsed_seconds"] if key in report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
