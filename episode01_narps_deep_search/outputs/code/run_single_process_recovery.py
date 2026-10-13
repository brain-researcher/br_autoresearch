#!/usr/bin/env python3
"""Necessary MV/MZ-at-SESOI recovery cells; never a complete gate verdict.

Only the seven frozen design columns of the preserved MC simulation are used.
Responses are freshly simulated and all four model/half fits use the unchanged
fixed-bound fitter. A representative replicate is a cost observation, not a
selected replacement branch or a qualified recovery probability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
from scipy.special import expit

import deadline_ddm
from hierarchical_behavior import fit_partially_pooled, load_split, make_data, parameter_names, trial_likelihood
from profile_recovery import truncated_normal
from qualify_assets import require_compute
from run_criterion_recovery import summarize_paired_scores as summarize_MC_scores


EPISODE = Path(__file__).resolve().parents[2]
DESIGN_SOURCE = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz"
IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
DESIGN_FIELDS = ("participant", "task_version", "run", "trial_order", "onset", "gain", "loss")
RESPONSE_FIELDS = ("choice", "no_response", "strength", "response_time", "response_category")
HALVES = (((1, 4), (2, 3)), ((2, 3), (1, 4)))
BASE_SEEDS = {"MV": 810031, "MZ": 910031}
REPORT_NAME = "single_process_recovery.json"


def cell_seed(model, index):
    if model not in BASE_SEEDS or type(index) is not int or not 0 <= index < 200:
        raise ValueError("Only MV/MZ fixed replicate indices 0 through 199 are defined")
    return BASE_SEEDS[model] + 10007 * index


def focal_truth(model):
    cell_seed(model, 0)
    return {"eta": math.log(1.15) if model == "MV" else 0.,
            "rho": math.log(.45 / .55) if model == "MZ" else 0., "criterion": 0.}


def project_design(header, values):
    """Access only whitelisted column positions, ignoring old synthetic outcomes."""
    if len(set(header)) != len(header) or not set(DESIGN_FIELDS).issubset(header):
        raise ValueError("Design header is incomplete or duplicated")
    return {name: values[header.index(name)] for name in DESIGN_FIELDS}


def load_design(path):
    if path.resolve() != DESIGN_SOURCE.resolve():
        raise ValueError("Only the preserved MC attempt003 simulation is a design source")
    source, slots, counts, versions = defaultdict(list), set(), Counter(), {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        for values in reader:
            if len(values) != len(header):
                raise ValueError("Design row width differs from its header")
            row = project_design(header, values)
            participant, version = row["participant"], row["task_version"]
            run, trial = int(row["run"]), int(row["trial_order"])
            key = participant, run, trial
            if not participant or key in slots or run not in (1, 2, 3, 4) or not 1 <= trial <= 64:
                raise ValueError("Duplicate or invalid participant/run/trial identity")
            if version not in ("EI", "ER") or versions.setdefault(participant, version) != version:
                raise ValueError("Invalid or changing task version")
            if not all(math.isfinite(float(row[name])) for name in ("gain", "loss", "onset")):
                raise ValueError("Nonfinite design value")
            if float(row["gain"]) <= 0 or float(row["loss"]) <= 0:
                raise ValueError("Offers must retain positive physical magnitudes")
            slots.add(key)
            counts[participant, run] += 1
            source[participant].append(row)
            if len(slots) > 27648:
                raise ValueError("Design exceeds the fixed trial support")
    if (len(slots) != 27648 or len(source) != 108 or len(counts) != 432
            or set(counts.values()) != {64} or Counter(versions.values()) != {"EI": 54, "ER": 54}):
        raise ValueError("Exact 108-participant/432-run/27648-trial EI/ER support is required")
    for participant, rows in source.items():
        gains = range(10, 41, 2) if versions[participant] == "EI" else range(5, 21)
        expected = {(float(gain), float(loss)) for gain in gains for loss in range(5, 21)}
        if {(float(row["gain"]), float(row["loss"])) for row in rows} != expected:
            raise ValueError("Participant does not preserve the full version-specific offer matrix")
    return dict(source)


def simulate_design(source, model, seed, spec, check=lambda: None):
    rng = np.random.default_rng(seed)
    truth, simulated = {}, []
    for participant in sorted(source):
        check()
        rows = source[participant]
        params = {}
        for name in ("kappa", "log_a", "nondecision_logit"):
            prior = spec["parameters"][name]
            params[name] = truncated_normal(rng, prior["population_center"],
                                           prior["individual_scale_center"], prior["bounds"])
        params.update(focal_truth(model))
        truth[participant] = params
        drift = deadline_ddm.drift_rate([float(row["gain"]) for row in rows],
            [float(row["loss"]) for row in rows], params["kappa"],
            eta=params["eta"], criterion=params["criterion"])
        choices, rt = deadline_ddm.simulate(rng, drift, np.exp(params["log_a"]), expit(params["rho"]),
            4 * expit(params["nondecision_logit"]), spec["contaminant"]["fraction"],
            spec["contaminant"]["miss_probability"], deadline=4.)
        if np.shape(choices) != (len(rows),) or np.shape(rt) != (len(rows),):
            raise ValueError("Simulator changed trial support")
        for row, choice, response_time in zip(rows, choices, rt):
            if choice not in (-1, 0, 1) or (choice == -1 and not np.isnan(response_time)):
                raise ValueError("Invalid simulated choice/censoring")
            if choice != -1 and not (math.isfinite(response_time) and 0 < response_time <= 4):
                raise ValueError("Invalid simulated response time")
            record = {name: row[name] for name in DESIGN_FIELDS}
            record.update(choice=int(choice), no_response=int(choice == -1), strength=-1,
                response_time="" if choice == -1 else float(response_time),
                response_category={-1: "no_response", 0: "simulated_reject", 1: "simulated_accept"}[int(choice)])
            simulated.append(record)
    return truth, simulated


def summarize_paired_scores(scores, model, seed):
    cell_seed(model, 0)
    # Reuse the existing statistic and bootstrap exactly; rename the focal key
    # only at the API boundary, without treating an MV/MZ fit as an MC fit.
    mapped = {participant: {"M0": row["M0"], "MC": row[model], "version": row["version"]}
              for participant, row in scores.items()}
    if not all(math.isfinite(row[name]) for row in mapped.values() for name in ("M0", "MC")):
        raise ValueError("Nonfinite held-out participant score")
    result = summarize_MC_scores(mapped, seed)
    requirements = result["requirements"]
    core = bool(requirements["mean_gain_at_least_0p01"]
                and requirements["mean_gain_greater_than_two_participant_SE"])
    result.update(focal_model=model, core_family_selection_pass=core,
        counts_as_curtailment_failure=not core,
        richer_primary_diagnostic_pass=result["necessary_primary_comparison_pass"],
        curtailment_rule="only_mean_at_least_0p01_and_mean_strictly_greater_than_2_participant_SE",
        diagnostics_not_recovery_rate_gates=["positive_in_both_task_versions", "positive_participant_median",
                                            "participant_median_95CI_lower_positive"])
    return result


def validate_fit(fit, model, training_runs, scoring_runs, spec, people):
    names = parameter_names(model, "fixed")
    if (fit.get("model") != model or fit.get("rival") != "fixed"
            or fit.get("training_runs") != list(training_runs) or fit.get("scoring_runs") != list(scoring_runs)
            or fit.get("participants") != sorted(people) or fit.get("parameter_names") != names
            or fit.get("version_order") != ["EI", "ER"] or fit.get("inference") != spec["method"]):
        raise ValueError("Fit model, split, participant or parameter identity changed")
    if fit.get("converged") is not True or fit.get("subject_optimizer_failures") != []:
        raise RuntimeError("Nonconverged fit is technical; it is not a recovery failure")
    theta = np.asarray(fit["parameters"], dtype=float)
    if theta.shape != (108, len(names)) or not np.isfinite(theta).all():
        raise ValueError("Incomplete or nonfinite participant parameters")
    for column, name in enumerate(names):
        low, high = spec["parameters"][name]["bounds"]
        if np.any(theta[:, column] < low) or np.any(theta[:, column] > high):
            raise ValueError("Fitted parameter outside unchanged support")
    for field, shape in (("global_mean", (len(names),)), ("version_means", (2, len(names))),
                         ("version_standard_deviations", (2, len(names)))):
        values = np.asarray(fit.get(field), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Incomplete or nonfinite fitted hyperparameters: {field}")
        if field == "version_standard_deviations" and np.any(values <= 0):
            raise ValueError("Nonpositive fitted population scale")
    objective = np.asarray(fit["objective_history"], dtype=float)
    if (objective.ndim != 1 or not 2 <= len(objective) <= spec["optimizer"]["maximum_sweeps"]
            or not np.isfinite(objective).all()
            or abs(objective[-1] - objective[-2]) > spec["optimizer"]["joint_relative_tolerance"] * (1 + abs(objective[-2]))):
        raise ValueError("Fit lacks the unchanged convergence evidence")
    if not isinstance(fit.get("support_contacts"), list):
        raise ValueError("Missing numeric-support diagnostics")


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Recovery interrupted by signal {signum}; incomplete replicate is technical")


def run(args):
    require_compute()
    seed = cell_seed(args.model, args.replicate_index)
    if not 1 <= args.max_seconds <= 5400:
        raise ValueError("Internal representative envelope must be 1 through 5400 seconds")
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged saved behavioral implementation")
    spec = json.loads(args.implementation.read_text())
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("The fixed implementation/recovery cell changed")
    output = args.output_dir.resolve()
    if output.exists() or DESIGN_SOURCE.parent == output or DESIGN_SOURCE.parent in output.parents:
        raise FileExistsError("Use a fresh output outside the preserved MC attempt")
    output.mkdir(parents=True)
    started = time.perf_counter()
    deadline = time.monotonic() + args.max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-replicate deadline exceeded")
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": f"{args.model}_SESOI_fitted_recovery_component",
        "generating_model": args.model, "replicate_index": args.replicate_index, "planned_cell_replicates": 200,
        "seed": seed, "SESOI_level": 1., "focal_generating_truth": focal_truth(args.model),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": spec, "design_source": str(args.design_events), "design_fields": list(DESIGN_FIELDS),
        "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
        "nuisance_generation": "recorded_center_and_individual_scale_truncated_normal_as_profile01",
        "deadline_missingness": "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel",
        "focal_effect_between_participant_variance": 0., "individual_rank_RMSE_qualification": "not_evaluated_constant_focal_effect",
        "full_behavioral_gate": "not_evaluated", "full_recovery_pass": None,
        "status": "running", "valid_recovery_component": False, "counts_as_curtailment_failure": False,
        "internal_seconds": args.max_seconds, "fits": []}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    try:
        write_report(output, report)
        source = load_design(args.design_events)
        truth, simulated = simulate_design(source, args.model, seed, spec, check)
        synthetic = output / "simulated_events.csv.gz"
        with gzip.open(synthetic, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DESIGN_FIELDS + RESPONSE_FIELDS)
            writer.writeheader()
            writer.writerows(simulated)
        (output / "generating_parameters.json").write_text(json.dumps(truth, indent=2, allow_nan=False) + "\n")
        report.update(participants=108, runs=432, trials=27648,
            simulation_seconds=time.perf_counter() - started,
            simulated_deadline_misses=sum(row["no_response"] for row in simulated))
        write_report(output, report)
        scores = {p: {"M0": 0., args.model: 0., "version": rows[0]["task_version"]} for p, rows in source.items()}
        for training_runs, scoring_runs in HALVES:
            check()
            training, scoring = load_split(synthetic, list(training_runs), list(scoring_runs))
            for model in ("M0", args.model):
                check()
                fit_started = time.perf_counter()
                fit = fit_partially_pooled(training, model, "fixed", spec)
                fit.update(training_runs=list(training_runs), scoring_runs=list(scoring_runs),
                           elapsed_seconds=time.perf_counter() - fit_started)
                json.dumps(fit, allow_nan=False)  # Do not poison terminal reporting with nonfinite fits.
                report["fits"].append(fit)
                write_report(output, report)  # Every returned complete fit precedes any held-out scoring.
                check()
                validate_fit(fit, model, training_runs, scoring_runs, spec, source)
                for participant, theta in zip(fit["participants"], fit["parameters"]):
                    check()
                    data = make_data(scoring[participant]["rows"])
                    logp = np.asarray(trial_likelihood(theta, fit["parameter_names"], data, spec, "fixed"))
                    if logp.shape != (128,) or not np.isfinite(logp).all():
                        raise ValueError("Held-out likelihood must contain 128 finite trial scores")
                    scores[participant][model] += float(logp.sum()) / 256
                print(json.dumps({"model": model, "training_runs": list(training_runs),
                    "converged": True, "elapsed_seconds": fit["elapsed_seconds"],
                    "support_contacts": len(fit["support_contacts"])}), flush=True)
        comparison = summarize_paired_scores(scores, args.model, seed + 4099)
        check()
        report.update(participant_scores=scores, comparison=comparison,
            valid_recovery_component=True, status="necessary_primary_comparison_completed",
            counts_as_curtailment_failure=comparison["counts_as_curtailment_failure"])
        write_report(output, report)
        check()
    except Exception as error:
        report.update(status="technical_recovery_failure", valid_recovery_component=False,
            counts_as_curtailment_failure=False, error=f"{type(error).__name__}: {error}")
        report.pop("comparison", None)
        report.pop("participant_scores", None)
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("MV", "MZ"), required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--design-events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "generating_model", "replicate_index",
        "valid_recovery_component", "counts_as_curtailment_failure", "comparison", "elapsed_seconds", "error")
        if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["valid_recovery_component"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
