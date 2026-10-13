#!/usr/bin/env python3
"""Resume complete synthetic MC recovery fits into a fresh attempt.

The coordinator must establish that the source writer has stopped and that
the fitter/likelihood code is unchanged. Old reports contain the numerical
specification, not a source-code snapshot. Only complete converged model/half
fits are reused; an interrupted optimizer is always started afresh.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from hierarchical_behavior import fit_partially_pooled, load_split, make_data, parameter_names, trial_likelihood
from qualify_assets import require_compute
from run_criterion_recovery import summarize_paired_scores


FIT_KEYS = [(model, training, scoring)
            for training, scoring in [((1, 4), (2, 3)), ((2, 3), (1, 4))]
            for model in ("M0", "MC")]
IDENTITY = {
    "analysis_id": "behavioral_mechanism_kill_test",
    "phase": "MC_SESOI_fitted_recovery_component",
    "generating_model": "MC", "SESOI_level": 1.,
    "calibration": "five_point_decrease_in_four_second_mixture_level_acceptance_at_zero_reference_drift",
    "nuisance_generation": "recorded_center_and_individual_scale_truncated_normal_as_profile01",
    "deadline_missingness": "generated_from_the_frozen_likelihood_all_original_trial_slots_retained",
    "observed_choices_or_RTs_used": False, "full_behavioral_gate": "not_evaluated",
}


def validate_report_identity(report, spec, replicate_index):
    count = spec["recovery_plan"]["replicates_per_qualified_generating_cell"]
    expected = dict(IDENTITY, implementation=spec, replicate_index=replicate_index,
                    planned_cell_replicates=count, seed=710031 + 10007*replicate_index)
    if not 0 <= replicate_index < count:
        raise ValueError("Replicate index is outside the recorded cell")
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Source report identity mismatch: {key}")
    if not isinstance(report.get("fits"), list) or not report["fits"]:
        raise ValueError("No complete saved fit is available to resume")


def validate_simulation(path, truth, report, spec):
    """Check saved synthetic support without opening observed source outcomes."""
    versions, slots, counts, misses = {}, set(), Counter(), 0
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            participant, version = row["participant"], row["task_version"]
            run, trial = int(row["run"]), int(row["trial_order"])
            key = participant, run, trial
            if key in slots or run not in (1, 2, 3, 4) or not 1 <= trial <= 64:
                raise ValueError("Duplicate or wrong participant/run/trial identity")
            slots.add(key)
            counts[participant, run] += 1
            if version not in ("EI", "ER") or versions.setdefault(participant, version) != version:
                raise ValueError("Invalid or changing participant task version")
            if not all(math.isfinite(float(row[name])) for name in ("gain", "loss", "onset")):
                raise ValueError("Nonfinite saved design covariate")
            choice = int(row["choice"])
            if choice not in (-1, 0, 1) or int(row["no_response"]) != int(choice == -1):
                raise ValueError("Invalid synthetic response/deadline coding")
            expected_category = {-1: "no_response", 0: "simulated_reject", 1: "simulated_accept"}[choice]
            if row["response_category"] != expected_category or int(row["strength"]) != -1:
                raise ValueError("Saved rows are not the selected synthetic response format")
            if choice == -1:
                if row["response_time"] != "":
                    raise ValueError("A synthetic deadline miss has a response time")
                misses += 1
            elif not 0 < float(row["response_time"]) <= 4:
                raise ValueError("Synthetic response time is outside the deadline")
    if (len(slots) != 27648 or len(versions) != 108
            or Counter(versions.values()) != {"EI": 54, "ER": 54}
            or len(counts) != 432 or set(counts.values()) != {64}):
        raise ValueError("Saved simulation does not retain the exact full design support")
    if misses != report.get("simulated_deadline_misses") or set(truth) != set(versions):
        raise ValueError("Saved simulation, truth and report disagree")
    for params in truth.values():
        if set(params) != {"kappa", "log_a", "nondecision_logit", "criterion"}:
            raise ValueError("Generating-parameter family differs from MC")
        for name, value in params.items():
            low, high = spec["parameters"][name]["bounds"]
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid saved generating parameter")
    return versions


def validate_fit(fit, spec, participants):
    key = fit.get("model"), tuple(fit.get("training_runs", [])), tuple(fit.get("scoring_runs", []))
    if key not in FIT_KEYS or fit.get("rival") != "fixed":
        raise ValueError("Wrong model, rival or primary half identity")
    if fit.get("converged") is not True or fit.get("subject_optimizer_failures") != []:
        raise ValueError("Only a fully converged fit may be reused")
    names = parameter_names(key[0], "fixed")
    if (fit.get("parameter_names") != names or fit.get("participants") != sorted(participants)
            or fit.get("version_order") != ["EI", "ER"]
            or fit.get("inference") != spec["method"]):
        raise ValueError("Fit parameter, participant or inference identity mismatch")
    parameters = np.asarray(fit.get("parameters"), dtype=float)
    if parameters.shape != (108, len(names)) or not np.isfinite(parameters).all():
        raise ValueError("Incomplete or nonfinite participant parameters")
    for column, name in enumerate(names):
        low, high = spec["parameters"][name]["bounds"]
        if np.any(parameters[:, column] < low) or np.any(parameters[:, column] > high):
            raise ValueError("Saved participant parameter is outside fixed support")
    for field, shape in [("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))]:
        values = np.asarray(fit.get(field), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Incomplete fitted hyperparameters: {field}")
        if field == "version_standard_deviations" and np.any(values <= 0):
            raise ValueError("Nonpositive fitted population scale")
    objectives = np.asarray(fit.get("objective_history"), dtype=float)
    if objectives.ndim != 1 or len(objectives) < 2 or not np.isfinite(objectives).all():
        raise ValueError("Missing complete convergence evidence")
    if (len(objectives) > spec["optimizer"]["maximum_sweeps"]
            or abs(objectives[-1] - objectives[-2]) > spec["optimizer"]["joint_relative_tolerance"]*(1+abs(objectives[-2]))):
        raise ValueError("Saved objective history fails the unchanged convergence rule")
    if not isinstance(fit.get("support_contacts"), list):
        raise ValueError("Saved fit lacks required support-contact diagnostics")
    return key


def collect_reusable_fits(report, spec, participants):
    reusable = {}
    for fit in report["fits"]:
        key = validate_fit(fit, spec, participants)
        if key in reusable:
            raise ValueError("Duplicate model/half checkpoint")
        reusable[key] = copy.deepcopy(fit)
    # A report produced by the original sequential runner can only be a prefix.
    if list(reusable) != FIT_KEYS[:len(reusable)]:
        raise ValueError("Saved fits are not the registered ordered model/half prefix")
    return reusable


def write_report(directory, report):
    temporary = directory / "criterion_recovery.json.partial"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "criterion_recovery.json")


def resume(args):
    require_compute()
    source = args.source_attempt.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents or output.exists():
        raise FileExistsError("Resume requires a fresh directory outside the preserved source attempt")
    spec = json.loads(args.implementation.read_text())
    prior = json.loads((source / "criterion_recovery.json").read_text())
    validate_report_identity(prior, spec, args.replicate_index)
    truth = json.loads((source / "generating_parameters.json").read_text())
    synthetic = source / "simulated_events.csv.gz"
    versions = validate_simulation(synthetic, truth, prior, spec)
    reusable = collect_reusable_fits(prior, spec, versions)
    output.mkdir(parents=True)
    # Both payloads come from this one stopped attempt; nothing is regenerated.
    for filename in ("simulated_events.csv.gz", "generating_parameters.json"):
        shutil.copyfile(source / filename, output / filename)
    synthetic = output / "simulated_events.csv.gz"
    started = time.perf_counter()
    report = {key: copy.deepcopy(prior[key]) for key in
              list(IDENTITY) + ["implementation", "replicate_index", "planned_cell_replicates", "seed",
                                "simulation_seconds", "simulated_deadline_misses"]}
    report.update(job_id=os.environ["SLURM_JOB_ID"], started_utc=datetime.now(timezone.utc).isoformat(),
                  fits=[], status="resuming_saved_converged_fits", valid_recovery_component=False,
                  resume_source_attempt=str(source), resume_source_job_id=prior.get("job_id"),
                  reused_fit_count=len(reusable), new_fit_count=0,
                  source_report_elapsed_seconds=prior.get("elapsed_seconds"),
                  source_completed_fit_seconds=sum(fit["elapsed_seconds"] for fit in reusable.values()),
                  implementation_identity_basis="exact_embedded_spec_and_coordinator_unchanged_code_boundary")
    scores = {participant: {"M0": 0., "MC": 0., "version": version}
              for participant, version in versions.items()}
    write_report(output, report)
    try:
        for model, training_runs, scoring_runs in FIT_KEYS:
            training, scoring = load_split(synthetic, list(training_runs), list(scoring_runs))
            key = model, training_runs, scoring_runs
            if key in reusable:
                fit = reusable[key]
            else:
                fit_started = time.perf_counter()
                fit = fit_partially_pooled(training, model, "fixed", spec)
                fit.update(training_runs=list(training_runs), scoring_runs=list(scoring_runs),
                           elapsed_seconds=time.perf_counter()-fit_started)
                report["new_fit_count"] += 1
            report["fits"].append(fit)
            write_report(output, report)
            validate_fit(fit, spec, versions)
            for participant, theta in zip(fit["participants"], fit["parameters"]):
                logp = trial_likelihood(theta, fit["parameter_names"], make_data(scoring[participant]["rows"]), spec, "fixed")
                if not np.isfinite(logp).all():
                    raise ValueError("Nonfinite recomputed synthetic score")
                scores[participant][model] += float(logp.sum())/256
            print(json.dumps({"model": model, "training_runs": training_runs,
                              "reused": key in reusable, "converged": True}), flush=True)
        report.update(participant_scores=scores, comparison=summarize_paired_scores(scores, report["seed"]+4099),
                      valid_recovery_component=True, status="necessary_primary_comparison_completed")
    except Exception as exc:
        report.update(status="technical_recovery_failure", valid_recovery_component=False,
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attempt", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps({key: value for key, value in resume(parser.parse_args(argv)).items()
                      if key in ("status", "comparison", "reused_fit_count", "new_fit_count", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
