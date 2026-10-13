#!/usr/bin/env python3
"""An explicitly selected indexed twice-MZ MVC numerical repair, never a retry dispatcher.

The reviewed pilot owns the timed two-fit execution loop. This adapter binds
actual indexed sources and checkpoint identities in this process and restores
every binding on return or exception. Existing files and reports stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import run_twice_mz_mvc_repair_pilot as engine
import run_twice_mz_rival_component as original_component


EPISODE, IMPLEMENTATION = engine.EPISODE, engine.IMPLEMENTATION
FAILED_ARRAY = 46638796
FAILED_ROOT = EPISODE / f"outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-menu-MVC-wave-{FAILED_ARRAY}"
REPORT_NAME = "twice_mz_mvc_repair_component.json"
PHASE = "MZ_twice_SESOI_indexed_MVC_runtime1500_numerical_repair"
MAX_SECONDS, THREAD_ENV = engine.MAX_SECONDS, engine.THREAD_ENV


def source_report(index):
    return original_component.source_report(index)


def failed_path(index):
    source_report(index)  # Original strict1..199 identity guard; no index0 route.
    return FAILED_ROOT / f"replicate{index}" / original_component.REPORT_NAME


def repair_context(index, failed, source, people, original, helpers):
    """Require this array's terminal first-half500 exhaustion, not status alone."""
    import summarize_twice_mz_menu_wave as collector

    path = failed_path(index)
    admitted, evidence = collector.validate_rivals(failed, source, index, original, ("MVC",))
    if (original["optimizer"]["maximum_sweeps"] != 500 or admitted
            or failed.get("status") != "technical_rival_component_failure"
            or failed.get("error") != engine.NONCONVERGENCE
            or failed.get("parallel_workers_cleaned_up") is not True or failed.get("parallel_failure") is not None
            or failed.get("source_baseline_validated") is not True or failed.get("saved_scores_replayed") is not True
            or failed.get("fresh_fit_calls") != 1 or failed.get("completed_rivals") != []
            or len(failed["fits"]) != 1 or failed.get("scored_components") != []
            or failed.get("scored_primary_directions") != []):
        raise ValueError("Only a terminal original500 first-half MVC exhaustion with clean workers is admissible")
    fit = failed["fits"][0]
    history = fit.get("objective_history")
    if (fit.get("model") != "MVC" or fit.get("rival") != "fixed" or fit.get("converged") is not False
            or fit.get("training_runs") != [1, 4] or fit.get("scoring_runs") != [2, 3]
            or fit.get("participants") != sorted(people) or fit.get("subject_optimizer_failures") != []
            or not isinstance(history, list) or len(history) != 500
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in history)):
        raise ValueError("Require the actual cohort's finite500-sweep forward fit without subject optimizer failures")
    json.dumps(fit, allow_nan=False)
    delta = abs(history[-1] - history[-2])
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if delta <= threshold:
        raise ValueError("The retained500th sweep must fail the unchanged convergence criterion")
    return {
        "kind": "maximum_sweeps_only", "classification": "first_half_500_sweep_exhaustion",
        "failed_report": str(path), "failed_array_job_id": FAILED_ARRAY,
        "failed_job_id": str(failed["job_id"]), "failed_finished_utc": failed["finished_utc"],
        "failed_error": failed["error"], "failed_source_finished_utc": failed["source_finished_utc"],
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"]], "previous_MVC_fit": deepcopy(fit),
        "previous_completed_rivals": [], "previous_scored_components": [],
        "previous_participant_scores": deepcopy(failed["participant_scores"]),
        "previous_terminal_evidence": evidence,
        "final_objective_change": delta, "unchanged_convergence_threshold": threshold,
        "original_maximum_sweeps": 500, "runtime_maximum_sweeps": 1500,
        "nonconverged_checkpoint_state_reused": False,
        "initialization_of_fresh_fits": "unchanged_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "cost_accounting": "Historical process and included fit durations are references only, not additive new charges.",
    }


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def run(args):
    index = args.replicate_index
    source, failed = source_report(index), failed_path(index)
    output = args.output_dir.resolve()
    protected = (FAILED_ROOT.resolve(), source_report(1).parent.parent.resolve(),
                 source_report(2).parent.parent.resolve())
    if any(output == root or root in output.parents for root in protected):
        raise FileExistsError("Preserve the full original failed array and both saved baseline arrays")

    def validate_source(report, truth, people, spec, helpers):
        original_component.validate_source(report, truth, people, spec, index, helpers)

    def selected_context(report, baseline, people, spec, helpers):
        return repair_context(index, report, baseline, people, spec, helpers)

    def checkpoint(output, report):
        # Identity is attached before every durable write, including failures
        # before loading the specification/source. No source report is relabeled.
        report.update(phase=PHASE, replicate_index=index, seed=1900057 + 10007 * index,
            selected_replicates=[index], component_scope="MVC_numerical_repair_only",
            component_execution_selected_by_coordinator=True, subsequent_wave_selected=False,
            qualification_boundary="Coordinator requires successful pilot qualification and separately selects exact terminal indices/resources; this component never dispatches or retries.",
            interpretation="One explicitly selected indexed twice-MZ MVC repair. Both halves start from unchanged priors; only the private sweep ceiling is1500. Original500 source and failure evidence remain separate; no full-menu, arbitration, rate, interval or recovery verdict.")
        report.pop("full_menu_wave_selected", None)
        write_report(output, report)

    bindings = {"EPISODE": EPISODE, "IMPLEMENTATION": IMPLEMENTATION, "SOURCE_REPORT": source,
        "FAILED_REPORT": failed, "PHASE": PHASE, "REPORT_NAME": REPORT_NAME,
        "repair_context": selected_context, "write_report": checkpoint}
    previous = {name: getattr(engine, name) for name in bindings}
    previous_validator = engine.pilot.validate_source
    try:
        for name, value in bindings.items():
            setattr(engine, name, value)
        engine.pilot.validate_source = validate_source
        return engine.run(args)
    finally:
        engine.pilot.validate_source = previous_validator
        for name, value in previous.items():
            setattr(engine, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "mvc_repair_complete", "replicate_index", "seed",
        "fresh_fit_calls", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["mvc_repair_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
