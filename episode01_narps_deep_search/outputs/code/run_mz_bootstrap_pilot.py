#!/usr/bin/env python3
"""One paired participant-refit MZ bootstrap draw for measured interval cost.

Only bootstrap index0 of the original terminal MZ replicate0 is selected.
The coordinator owns the completed point-bias prerequisite and launch decision.
This program does not select B, compute intervals, score models, simulate new
responses, retry fits, or make recovery/coverage claims. Source files are read
unchanged; duplicated participants exist only in memory with distinct IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

# Direct CLI execution fixes numeric workers before importing NumPy/SciPy.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

import numpy as np
from scipy.special import expit

from hierarchical_behavior import fit_partially_pooled, load_split
from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION, validate_fit
from summarize_mz_group_recovery import summarize_group_reports


EPISODE = Path(__file__).resolve().parents[2]
SOURCE_REPORT = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json"
SEED_COMPONENTS = (1200037, 910031, 0)
REPORT_NAME = "mz_bootstrap_pilot.json"
OUTPUT_LIMIT_BYTES = 20 * 1024 * 1024


def paired_draw(versions):
    """One fixed EI-then-ER draw from sorted original participant IDs."""
    if Counter(versions.values()) != {"EI": 54, "ER": 54} or len(versions) != 108:
        raise ValueError("The bootstrap requires all54 EI and54 ER participants")
    rng = np.random.default_rng(np.random.SeedSequence(SEED_COMPONENTS))
    draws = []
    for version in ("EI", "ER"):
        original = sorted(p for p, group in versions.items() if group == version)
        for slot, selected in enumerate(rng.integers(0, 54, size=54)):
            draws.append({"bootstrap_participant": f"bootstrap000-{version}-{slot:03}",
                          "source_participant": original[int(selected)],
                          "task_version": version, "stratum_draw_index": slot})
    return draws


def validate_source_halves(halves, versions):
    """Check the original synthetic cohort's intact run/trial/censoring support."""
    if len(halves) != 2:
        raise ValueError("Both primary source halves are required")
    for half, (training_runs, _) in zip(halves, HALVES):
        if set(half) != set(versions):
            raise ValueError("Synthetic events and terminal report have different participants")
        for participant, group in half.items():
            rows = group["rows"]
            if group["version"] != versions[participant] or len(rows) != 128:
                raise ValueError("Changed version or trial support in synthetic source")
            coordinates = set()
            for row in rows:
                run, trial = int(row["run"]), int(row["trial_order"])
                if (row["participant"] != participant or row["task_version"] != versions[participant]
                        or run not in training_runs or not 1 <= trial <= 64
                        or (run, trial) in coordinates):
                    raise ValueError("Duplicate or changed participant/run/trial identity")
                coordinates.add((run, trial))
                if not all(math.isfinite(float(row[name])) and float(row[name]) > 0 for name in ("gain", "loss")):
                    raise ValueError("Invalid physical offer in original synthetic source")
                choice, missed = int(row["choice"]), int(row["no_response"])
                if choice == -1:
                    if missed != 1 or row["response_time"] != "":
                        raise ValueError("Synthetic deadline miss has changed censoring support")
                elif choice in (0, 1):
                    if missed != 0 or not 0 < float(row["response_time"]) <= 4:
                        raise ValueError("Invalid original synthetic response time")
                else:
                    raise ValueError("Invalid original synthetic choice")
            if coordinates != {(run, trial) for run in training_runs for trial in range(1, 65)}:
                raise ValueError("Incomplete original two-run source support")


def resampled_half(half, draws):
    """Each draw is one whole cluster, including repeated source participants."""
    result = {}
    for draw in draws:
        pseudo = draw["bootstrap_participant"]
        original = half[draw["source_participant"]]
        if pseudo in result or original["version"] != draw["task_version"]:
            raise ValueError("Duplicate bootstrap identity or changed task version")
        result[pseudo] = {"version": original["version"],
                          "rows": [dict(row, participant=pseudo) for row in original["rows"]]}
    if len(result) != 108 or Counter(group["version"] for group in result.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Resampling changed the registered cohort size")
    return result


def physical_group_effects(fits, draws):
    versions = {draw["bootstrap_participant"]: draw["task_version"] for draw in draws}
    values = {"EI": [], "ER": []}
    for fit in fits:
        column = fit["parameter_names"].index("rho")
        for participant, theta in zip(fit["participants"], fit["parameters"]):
            values[versions[participant]].append(float(expit(theta[column])))
    if any(len(group) != 108 for group in values.values()):
        raise ValueError("Physical effect requires54 complete participants in each of both halves")
    return {version: .5 - math.fsum(values[version]) / 108 for version in ("EI", "ER")}


def write_report(output, report):
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > OUTPUT_LIMIT_BYTES:
        raise ValueError("Bootstrap pilot output exceeds20MiB")
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Bootstrap resource pilot interrupted by signal{signum}; incomplete draw is technical")


def run(args):
    require_compute()
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The internal bootstrap pilot limit must be1 through5400 seconds")
    if args.source_report.resolve() != SOURCE_REPORT.resolve():
        raise ValueError("Only the original fixed MZ replicate0 attempt001 is selected")
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged saved behavioral implementation")
    output = args.output_dir.resolve()
    if output.exists() or output == SOURCE_REPORT.parent or SOURCE_REPORT.parent in output.parents:
        raise FileExistsError("Select a fresh output outside the preserved source attempt")
    spec = json.loads(args.implementation.read_text())
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal bootstrap pilot deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_participant_bootstrap_resource_pilot",
              "source_report": str(SOURCE_REPORT), "source_replicate_index": 0, "source_seed": 910031,
              "bootstrap_index": 0, "seed_sequence_components": list(SEED_COMPONENTS),
              "generator": "numpy.default_rng(SeedSequence); EI then ER; sorted source IDs;54 integer draws per version",
              "implementation": spec, "internal_seconds": args.max_seconds,
              "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
              "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "bootstrap_draw_complete": False, "fits": [],
              "observed_choices_or_RTs_used": False, "new_responses_simulated": False,
              "heldout_scoring_performed": False, "source_files_modified": False,
              "bootstrap_draw_count_selected": 1, "interval_bootstrap_count_selected": None,
              "interval_convention_selected": None, "interval_coverage_evaluated": False,
              "group_interval_coverage_qualified": False, "full_recovery_gate": "not_evaluated"}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    try:
        write_report(output, report)
        baseline = json.loads(args.source_report.read_text())
        summary = summarize_group_reports([(args.source_report, baseline)], spec)
        if (baseline["replicate_index"] != 0 or baseline["seed"] != 910031
                or summary["valid_replicates"] != 1 or not baseline["valid_recovery_component"]):
            raise ValueError("A completed valid original MZ replicate0 is required")
        versions = {person: record["version"] for person, record in baseline["participant_scores"].items()}
        del summary, baseline
        check()
        source_events = SOURCE_REPORT.parent / "simulated_events.csv.gz"
        halves = load_split(source_events, [1, 4], [2, 3])
        validate_source_halves(halves, versions)
        draws = paired_draw(versions)
        report.update(source_events=str(source_events), participant_draws=draws,
                      same_draw_in_both_halves=True, bootstrap_participants=108,
                      unique_source_participants_by_version={v: len({d["source_participant"] for d in draws if d["task_version"] == v}) for v in ("EI", "ER")},
                      source_loading_and_draw_seconds=time.perf_counter() - started)
        write_report(output, report)
        for half, (training_runs, scoring_runs) in zip(halves, HALVES):
            check()
            training = resampled_half(half, draws)
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, "MZ", "fixed", spec)
            fit.update(training_runs=list(training_runs), scoring_runs=list(scoring_runs),
                       elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            # Preserve returned full fits for diagnosis, without declaring the
            # draw complete before both pass the unchanged fitter checks.
            write_report(output, report)
            check()
            validate_fit(fit, "MZ", training_runs, scoring_runs, spec, training)
            if (fit.get("group_interval_coverage_qualified") is not False
                    or fit.get("individual_rank_recovery_qualified") is not False):
                raise ValueError("A point fit cannot claim interval or individual recovery qualification")
            del training
            print(json.dumps({"bootstrap_index": 0, "training_runs": list(training_runs),
                              "fit_complete": True, "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        check()
        effects = physical_group_effects(report["fits"], draws)
        check()
        report.update(status="paired_refit_draw_completed", bootstrap_draw_complete=True,
                      physical_rejection_displacement=effects,
                      total_fit_seconds=math.fsum(fit["elapsed_seconds"] for fit in report["fits"]))
    except Exception as error:
        report.update(status="technical_bootstrap_pilot_failure", bootstrap_draw_complete=False,
                      error=f"{type(error).__name__}: {error}")
        report.pop("physical_rejection_displacement", None)
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if report["bootstrap_draw_complete"] and time.monotonic() >= deadline:
            report.update(status="technical_bootstrap_pilot_failure", bootstrap_draw_complete=False,
                          error="RuntimeError: Internal bootstrap pilot deadline exceeded during finalization")
            report.pop("physical_rejection_displacement", None)
        report.update(elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({name: report[name] for name in ("status", "bootstrap_draw_complete", "elapsed_seconds",
        "physical_rejection_displacement", "error") if name in report}, indent=2))
    return 0 if report["bootstrap_draw_complete"] and report["status"] == "paired_refit_draw_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
