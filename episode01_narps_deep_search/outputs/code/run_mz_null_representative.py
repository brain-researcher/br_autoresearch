#!/usr/bin/env python3
"""One fixed M0-null cohort with M0/MZ fits; no false-selection gate verdict.

This prospective representative measures cost and one nested diagnostic. It
does not select a 200-replicate wave, arbitrate the full menu, estimate a
false-selection rate, or evaluate group intervals. Preserve its simulation
and returned fits for possible later completion of the same cohort's menu.
"""
from __future__ import annotations

import argparse
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
from hierarchical_behavior import fit_partially_pooled, load_split
from profile_recovery import truncated_normal
from qualify_assets import require_compute
from run_single_process_recovery import (
    DESIGN_FIELDS, RESPONSE_FIELDS, DESIGN_SOURCE, HALVES, IMPLEMENTATION,
    load_design, validate_fit,
)
from run_single_process_rivals import score_fit
from summarize_mz_fixed_menu import nested_contrast, score_panel


SEED = 1300037
REPLICATE_INDEX = 0
GENERATING_MODEL = "M0"
FITTED_MODELS = ("M0", "MZ")
NULL_TRUTH = {"eta": 0., "rho": 0., "criterion": 0.}
REPORT_NAME = "mz_null_representative.json"


def simulate_null(source, spec, check=lambda: None):
    """Keep the established nuisance regime while making all focal terms null."""
    rng = np.random.default_rng(SEED)
    truth, simulated = {}, []
    for participant in sorted(source):
        check()
        rows = source[participant]
        params = {}
        for name in ("kappa", "log_a", "nondecision_logit"):
            prior = spec["parameters"][name]
            params[name] = truncated_normal(rng, prior["population_center"],
                                           prior["individual_scale_center"], prior["bounds"])
        params.update(NULL_TRUTH)
        truth[participant] = params
        drift = deadline_ddm.drift_rate(
            [float(row["gain"]) for row in rows], [float(row["loss"]) for row in rows],
            params["kappa"], eta=params["eta"], criterion=params["criterion"])
        choices, rt = deadline_ddm.simulate(
            rng, drift, np.exp(params["log_a"]), expit(params["rho"]),
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
                          response_category={-1: "no_response", 0: "simulated_reject",
                                             1: "simulated_accept"}[int(choice)])
            simulated.append(record)
    return truth, simulated


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Null representative interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if (args.design_events.resolve() != DESIGN_SOURCE.resolve()
            or args.implementation.resolve() != IMPLEMENTATION.resolve()):
        raise ValueError("Use the exact preserved design source and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The whole-process internal cap must be 1 through 5400 seconds")
    if any(os.environ.get(name) != "1" for name in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("All four numerical thread limits must be explicitly one")
    spec = json.loads(args.implementation.read_text())
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("The frozen implementation changed")
    output = args.output_dir.resolve()
    if output.exists() or output == DESIGN_SOURCE.parent or DESIGN_SOURCE.parent in output.parents:
        raise FileExistsError("Use a fresh output outside the preserved design source")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": "M0_null_MZ_pairwise_representative",
        "generating_model": GENERATING_MODEL, "fitted_models": list(FITTED_MODELS),
        "replicate_index": REPLICATE_INDEX, "seed": SEED, "SESOI_level": 0.,
        "focal_generating_truth": dict(NULL_TRUTH), "job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "implementation": spec,
        "design_source": str(DESIGN_SOURCE), "design_fields": list(DESIGN_FIELDS),
        "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
        "nuisance_generation": "recorded_center_and_individual_scale_truncated_normal_as_profile01",
        "deadline_missingness": "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel",
        "focal_effect_between_participant_variance": 0.,
        "individual_rank_RMSE_qualification": "not_evaluated_constant_focal_effect",
        "selected_replicates": [REPLICATE_INDEX], "wave_selected": False,
        "family_selection": "not_evaluated", "full_menu_false_selection": "not_evaluated",
        "false_selection_rate": "not_evaluated", "confusion_rate": "not_evaluated",
        "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated", "full_gate": "not_evaluated",
        "full_recovery_pass": None, "episode_terminal": False,
        "interpretation": "One M0-null representative and MZ-M0 nested diagnostic only. Pairwise qualification is not a full-menu false selection or an early-kill decision; no rate or interval verdict is computed.",
        "internal_seconds": args.max_seconds, "status": "running", "pairwise_diagnostic_complete": False,
        "fits": [], "scored_primary_directions": [],
    }
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        write_report(output, report)
        source = load_design(args.design_events)
        check()
        truth, simulated = simulate_null(source, spec, check)
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
        scores = {person: {"M0": 0., "MZ": 0., "version": rows[0]["task_version"]}
                  for person, rows in source.items()}
        for train, held in HALVES:
            check()
            training, scoring = load_split(synthetic, list(train), list(held))
            for model in FITTED_MODELS:
                check()
                fit_started = time.perf_counter()
                fit = fit_partially_pooled(training, model, "fixed", spec)
                fit.update(training_runs=list(train), scoring_runs=list(held),
                           elapsed_seconds=time.perf_counter() - fit_started)
                json.dumps(fit, allow_nan=False)
                report["fits"].append(fit)
                write_report(output, report)
                check()
                validate_fit(fit, model, train, held, spec, source)
                for person, value in score_fit(fit, scoring, spec, check).items():
                    scores[person][model] += value
                check()
                print(json.dumps({"model": model, "training_runs": list(train), "scoring_runs": list(held),
                                  "fit_completed": True, "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
        score_panel({"participant_scores": scores}, FITTED_MODELS)
        diagnostic = nested_contrast(scores, "MZ", "M0")
        check()
        report.update(participant_scores=scores, nested_comparison=diagnostic)
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_representative_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if succeeded and time.monotonic() >= deadline:
            succeeded = False
            report.update(status="technical_representative_failure", error="RuntimeError: Internal deadline reached during finalization")
        if not succeeded:
            report.pop("participant_scores", None)
            report.pop("nested_comparison", None)
        report.update(status="pairwise_diagnostic_completed" if succeeded else report["status"],
                      pairwise_diagnostic_complete=succeeded, elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in
                     ("status", "generating_model", "replicate_index", "seed", "pairwise_diagnostic_complete",
                      "nested_comparison", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["pairwise_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
