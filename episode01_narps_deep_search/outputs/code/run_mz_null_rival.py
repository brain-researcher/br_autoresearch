#!/usr/bin/env python3
"""Fit one rival on the completed, unchanged M0-null representative 46574266.

Replay the saved M0/MZ scores, then fit only the selected MC/MV/MVZ/MVC model
in both primary directions. No simulation, baseline refit, family selection,
false-selection rate, interval verdict, retry, or null wave is performed.
The coordinator establishes terminal process state separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

from hierarchical_behavior import fit_partially_pooled, load_split
from qualify_assets import require_compute
from run_single_process_recovery import DESIGN_FIELDS, DESIGN_SOURCE, HALVES, IMPLEMENTATION, validate_fit
from run_mz_saved_replicate_rival import MODELS, paired_diagnostics
from summarize_mz_fixed_menu import score_panel
import run_single_process_rivals as shared


SOURCE = Path(__file__).resolve().parents[2] / "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-46574266"
SOURCE_REPORT = "mz_null_representative.json"
SOURCE_JOB_ID = "46574266"
SEED = 1300037
REPLICATE_INDEX = 0
REPORT_NAME = "mz_null_rival.json"
CLAIM_FIELDS = ("family_selection", "full_menu_false_selection", "false_selection_rate", "confusion_rate",
                "group_recovery", "interval_coverage", "full_gate")


def validate_source(report, truth, people, spec):
    """Validate this null pilot without importing MZ-SESOI recovery semantics."""
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != "M0_null_MZ_pairwise_representative"
            or report.get("generating_model") != "M0" or report.get("fitted_models") != ["M0", "MZ"]
            or type(report.get("seed")) is not int or report["seed"] != SEED
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != REPLICATE_INDEX
            or str(report.get("job_id")) != SOURCE_JOB_ID or report.get("SESOI_level") != 0.
            or report.get("focal_generating_truth") != {"eta": 0., "rho": 0., "criterion": 0.}
            or report.get("implementation") != spec):
        raise ValueError("Only the fixed M0-null pilot identity and unchanged implementation are allowed")
    if (report.get("status") != "pairwise_diagnostic_completed"
            or report.get("pairwise_diagnostic_complete") is not True
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES]
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
        raise ValueError("The null representative must have completed both directions on its full cohort")
    for field in ("started_utc", "finished_utc"):
        value = report.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("The source report requires explicit dated terminal provenance")
    if (report.get("observed_choices_or_RTs_used") is not False
            or report.get("old_synthetic_responses_used") is not False
            or report.get("design_fields") != list(DESIGN_FIELDS)
            or Path(report.get("design_source", "")).resolve() != DESIGN_SOURCE.resolve()
            or report.get("nuisance_generation") != "recorded_center_and_individual_scale_truncated_normal_as_profile01"
            or report.get("deadline_missingness") != "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel"
            or report.get("focal_effect_between_participant_variance") != 0.):
        raise ValueError("The null source design, generation regime or response scope changed")
    if (any(report.get(field) != "not_evaluated" for field in CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or report.get("selected_replicates") != [REPLICATE_INDEX] or report.get("wave_selected") is not False):
        raise ValueError("The source must retain its single-representative diagnostic-only claim boundary")
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("The frozen implementation changed")
    if set(truth) != set(people):
        raise ValueError("Generating truth participant identities changed")
    for params in truth.values():
        if (set(params) != {"kappa", "log_a", "nondecision_logit", "eta", "rho", "criterion"}
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in params.values())
                or any(params[key] != 0. for key in ("eta", "rho", "criterion"))):
            raise ValueError("Saved generating parameters must retain exactly the finite M0-null truth")
        for name in ("kappa", "log_a", "nondecision_logit"):
            low, high = spec["parameters"][name]["bounds"]
            if not low <= params[name] <= high:
                raise ValueError("Saved nuisance truth is outside the frozen support")
    if report.get("simulated_deadline_misses") != sum(int(row["no_response"]) for rows in people.values() for row in rows):
        raise ValueError("Saved cohort deadline-miss accounting changed")
    scores = score_panel(report, ("M0", "MZ"))
    if set(scores) != set(people) or any(scores[person]["version"] != people[person][0]["task_version"] for person in people):
        raise ValueError("Saved scores and cohort participant/task-version identities differ")
    expected = [(model, train, held) for train, held in HALVES for model in ("M0", "MZ")]
    if len(report.get("fits", [])) != len(expected):
        raise ValueError("Four completed original M0/MZ fits are required")
    for fit, (model, train, held) in zip(report["fits"], expected):
        validate_fit(fit, model, train, held, spec, people)


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Null rival interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.model not in MODELS:
        raise ValueError("Select exactly one MC/MV/MVZ/MVC rival")
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the exact null pilot directory and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The whole-process internal cap must be 1 through 5400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Require one allocated CPU and all four numerical thread limits explicitly one")
    output = args.output_dir.resolve()
    if output.exists() or output == SOURCE.resolve() or SOURCE.resolve() in output.parents:
        raise FileExistsError("Use a fresh destination outside the preserved null cohort")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": "M0_null_saved_representative_rival_diagnostics",
        "generating_model": "M0", "fitted_model": args.model, "replicate_index": REPLICATE_INDEX,
        "seed": SEED, "source_dir": str(SOURCE), "source_report": SOURCE_REPORT, "source_job_id": SOURCE_JOB_ID,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "rival_diagnostics_complete": False,
        "saved_scores_replayed": False, "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
        "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS},
        "full_recovery_pass": None, "episode_terminal": False, "wave_selected": False,
        "interpretation": "One rival on the same fixed M0-null representative, with paired descriptive contrasts only. No full-menu false selection, family arbitration, rate, interval or gate verdict is computed.",
        "process_state_boundary": "The coordinator separately establishes that the source pilot is terminal; report completion fields are not scheduler evidence.",
        "fits": [], "scored_primary_directions": [], "completed_rivals": [], "diagnostics": {},
    }
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        write_report(output, report)
        spec = json.loads(args.implementation.read_text())
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        validate_source(source, truth, people, spec)
        check()
        splits = [load_split(events, list(train), list(held)) for train, held in HALVES]
        scores, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(implementation=spec, source_finished_utc=source["finished_utc"], saved_scores_replayed=True,
                      saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE,
                      participants=108, runs=432, trials=27648)
        write_report(output, report)
        for row in scores.values():
            row[args.model] = 0.
        for (train, held), (training, scoring) in zip(HALVES, splits):
            check()
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, args.model, "fixed", spec)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            write_report(output, report)
            check()
            validate_fit(fit, args.model, train, held, spec, people)
            for person, value in shared.score_fit(fit, scoring, spec, check).items():
                scores[person][args.model] += value
            check()
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
            print(json.dumps({"model": args.model, "replicate_index": REPLICATE_INDEX,
                              "training_runs": list(train), "fit_completed": True,
                              "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        score_panel({"participant_scores": scores}, ("M0", "MZ", args.model))
        diagnostics = paired_diagnostics(scores, args.model)
        check()
        report.update(participant_scores=scores, diagnostics={args.model: diagnostics}, completed_rivals=[args.model])
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_rival_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if succeeded and time.monotonic() >= deadline:
            succeeded = False
            report.update(status="technical_rival_failure", error="RuntimeError: Internal deadline reached during finalization")
        if not succeeded:
            report.pop("participant_scores", None)
            report.update(completed_rivals=[], diagnostics={})
        report.update(status="rival_diagnostics_completed" if succeeded else report["status"],
                      rival_diagnostics_complete=succeeded, elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in
                     ("status", "fitted_model", "replicate_index", "seed", "rival_diagnostics_complete",
                      "diagnostics", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
