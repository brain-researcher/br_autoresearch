#!/usr/bin/env python3
"""Collect explicitly terminal twice-SESOI fixed menus without selecting a family.

The coordinator supplies terminal selections; this program never scans, polls,
rescores events, or chooses an attempt. All 200 cohort indices stay visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from qualify_assets import require_compute
import run_twice_mz_menu_pilot as pilot
import summarize_mz_fixed_menu as fixed
import summarize_mz_twice_wave as twice


IMPLEMENTATION = twice.IMPLEMENTATION
PHASE = "MZ_twice_SESOI_fixed200_menu_nested_diagnostics"
PILOT_JOB = 46634879
PILOT_REPORT = twice.RUNS / f"20261005-MZ-twice-menu-pilot-{PILOT_JOB}-attempt001" / pilot.REPORT_NAME
COMPONENT_PHASE = "MZ_twice_SESOI_fixed_rival_component"
COMPONENT_REPORT = "twice_mz_rival_component.json"
CLAIMS = ("family_selection", "family_recovery_rate", "confusion_rate", "group_recovery", "interval_coverage", "full_gate")
MAX_SECONDS = 240
THREAD_ENV = pilot.THREAD_ENV


def component_array(value):
    """One model/array and its explicitly terminal indices, never a wildcard."""
    fields = value.split(":")
    if len(fields) != 3 or fields[0] not in pilot.RIVALS:
        raise argparse.ArgumentTypeError("Use MODEL:ARRAY_ID:INDEX,INDEX with MC, MV, MVZ or MVC")
    model, array, indices = fields[0], twice.array_id(fields[1]), twice.terminal_indices(fields[2])
    if 0 in indices:
        raise argparse.ArgumentTypeError("Index0 is bound to the original pilot, never an indexed component")
    return model, array, indices


def selection(indices, declarations, spec):
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < 200 for index in indices)):
        raise ValueError("Select unique explicit terminal cohort indices0 through199")
    # Reuse the existing spec check with an empty input selection: no reports
    # are read, no baseline outcomes are counted and no criterion is adopted.
    twice.pool([], twice.FIRST_ARRAY, twice.REMAINING_ARRAY, spec)
    chosen = {}
    for model, array, cohort_indices in declarations:
        if (model not in pilot.RIVALS or type(array) is not int or array < 1
                or not cohort_indices or len(set(cohort_indices)) != len(cohort_indices)
                or any(type(index) is not int or not 1 <= index < 200 or index not in indices for index in cohort_indices)):
            raise ValueError("Component declarations require selected1..199 indices and a positive explicit array")
        for index in cohort_indices:
            key = index, model
            if key in chosen:
                raise ValueError("A model/cohort may have exactly one selected terminal attempt")
            chosen[key] = array
    return chosen


def baseline_path(index):
    return twice.report_path(index, twice.FIRST_ARRAY, twice.REMAINING_ARRAY)


def component_path(index, model, array):
    return twice.RUNS / f"20261005-MZ-twice-menu-{model}-wave-{array}" / f"replicate{index}" / COMPONENT_REPORT


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Attempt and fit durations must be finite nonnegative numbers")
    return value


def terminal(report):
    try:
        start, finish = (datetime.fromisoformat(report[field]) for field in ("started_utc", "finished_utc"))
        if start.utcoffset() is None or finish.utcoffset() is None or finish < start:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Retain terminal timestamps") from error
    job = str(report.get("job_id", ""))
    if not job.isdecimal() or int(job) < 1:
        raise ValueError("Retain the actual positive worker job identity")
    return {"job_id": job, "started_utc": report["started_utc"], "finished_utc": report["finished_utc"],
            "elapsed_seconds": duration(report.get("elapsed_seconds")), "status": report.get("status")}


def validate_rivals(report, base, index, spec, models, *, is_pilot=False):
    """Admit only explicitly completed, validated pairs; retain technical prefixes."""
    metadata = terminal(report)
    phase = pilot.PHASE if is_pilot else COMPONENT_PHASE
    complete_key = "menu_pilot_complete" if is_pilot else "rival_component_complete"
    complete_status = "menu_pilot_completed" if is_pilot else "rival_component_completed"
    technical_status = "technical_menu_pilot_failure" if is_pilot else "technical_rival_component_failure"
    technical = report.get("status") == technical_status
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != phase
            or report.get("generating_model") != "MZ" or report.get("effect_level") != "twice"
            or report.get("SESOI_level") != 2. or report.get("generating_z") != .40
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != twice.levels.level_seed("twice", index)
            or report.get("selected_replicates") != [index]
            or report.get("source_report") != str(baseline_path(index))
            or report.get("source_dir") != str(baseline_path(index).parent)
            or (is_pilot and (index != 0 or metadata["job_id"] != str(PILOT_JOB)))
            or (not is_pilot and (index == 0 or report.get("fitted_model") != models[0]))):
        raise ValueError("Preserve the actual twice-cell phase, index, source, seed and model identity")
    if (any(report.get(field) != "not_evaluated" for field in CLAIMS)
            or report.get("full_recovery_pass") is not None
            or any(report.get(field) is not False for field in ("episode_terminal", "simulation_regenerated",
                "saved_M0_MZ_refitted", "saved_fit_parameters_reused", "observed_choices_or_RTs_used",
                "source_files_modified", "automatic_retry"))
            or (is_pilot and report.get("full_menu_wave_selected") is not False)
            or ("full_menu_wave_selected" in report and report["full_menu_wave_selected"] is not False)
            or report.get("workers") != 4 or report.get("allocated_cpus") != 4
            or report.get("thread_environment") != {name: "1" for name in THREAD_ENV}
            or report.get("original_maximum_sweeps") != 500
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= 5400):
        raise ValueError("Preserve original500 execution and diagnostic-only claims")
    if not is_pilot and (report.get("component_execution_selected_by_coordinator") is not True
            or report.get("subsequent_wave_selected") is not False or report.get("automatic_resume") is not False):
        raise ValueError("Only an explicitly selected component, with no automatic follow-on, may enter")
    if technical:
        if report.get(complete_key) is not False or not isinstance(report.get("error"), str) or not report["error"]:
            raise ValueError("Technical attempts require an error and no complete-process claim")
        metadata["error"] = report["error"]
    elif report.get("status") != complete_status or report.get(complete_key) is not True:
        raise ValueError("The selected report must be completed or an identified terminal technical failure")
    diagnostics = ("nested_comparisons", "nested_comparison_eligibility", "eligible_models_diagnostic",
                   "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models")
    if technical and any(field in report for field in diagnostics):
        raise ValueError("Technical attempts cannot retain a full-menu diagnostic claim")
    fits, scored, completed = (report.get(field) for field in ("fits", "scored_components", "completed_rivals"))
    expected = [(model, direction, train, held) for model in models
                for direction, (train, held) in enumerate(twice.core.HALVES)]
    if (not isinstance(fits, list) or len(fits) > len(expected) or not isinstance(scored, list)
            or not 0 <= len(fits) - len(scored) <= 1 or not isinstance(completed, list)
            or completed != list(models[:len(completed)]) or len(completed) > len(models)
            or not 2 * len(completed) <= len(scored) <= 2 * (len(completed) + 1)):
        raise ValueError("Returned/scored fits and completed models must retain their ordered prefixes")
    if (type(report.get("fresh_fit_calls")) is not int
            or not len(fits) <= report["fresh_fit_calls"] <= min(len(expected), len(fits) + 1)):
        raise ValueError("Fresh-fit call count must agree with the retained sequential prefix")
    if not is_pilot and report.get("scored_primary_directions") != [
            {"training_runs": list(train), "scoring_runs": list(held)} for _, _, train, held in expected[:len(scored)]]:
        raise ValueError("Scored primary directions must agree with contribution records")
    fit_fields = ("model", "training_runs", "scoring_runs", "converged", "elapsed_seconds",
                  "support_contacts", "subject_optimizer_failures")
    metadata.update(completed_rivals=list(completed),
        returned_fits=[{**{key: deepcopy(fit.get(key)) for key in fit_fields},
                       "sweeps": len(fit["objective_history"]) if isinstance(fit.get("objective_history"), list) else None}
                      for fit in fits],
        scored_components=[{key: deepcopy(value) for key, value in row.items() if key != "participant_contributions"} for row in scored],
        parallel_failure=deepcopy(report.get("parallel_failure")),
        parallel_workers_cleaned_up=report.get("parallel_workers_cleaned_up"))
    validated, replayed = report.get("source_baseline_validated"), report.get("saved_scores_replayed")
    if type(validated) is not bool or type(replayed) is not bool:
        raise ValueError("Source validation and replay states must be explicit")
    if "implementation" in report and report["implementation"] != spec:
        raise ValueError("Original implementation changed")
    if not validated:
        if not technical or replayed or fits or scored or completed or "participant_scores" in report:
            raise ValueError("An unvalidated source cannot supply fits or scores")
        return {}, metadata
    if (report.get("implementation") != spec or report.get("source_finished_utc") != base["finished_utc"]
            or (not is_pilot and report.get("source_job_id") != str(base["job_id"]))
            or report.get("focal_generating_truth") != base["focal_generating_truth"]
            or [report.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]):
        raise ValueError("Validated source metadata must match the original baseline completion and support")
    if not replayed:
        if not technical or fits or scored or completed or "participant_scores" in report:
            raise ValueError("No new fitted or scored component is admissible before saved-score replay")
        return {}, metadata
    if (report.get("saved_score_replay_absolute_tolerance") != 1e-12
            or duration(report.get("saved_score_replay_max_absolute_discrepancy")) > 1e-12):
        raise ValueError("Require the unchanged successful1e-12 original-score replay")
    baseline = fixed.score_panel(base, ("M0", "MZ"))
    panel = fixed.score_panel(report, ("M0", "MZ", *completed))
    if set(panel) != set(baseline) or any(panel[p][k] != baseline[p][k] for p in baseline for k in ("version", "M0", "MZ")):
        raise ValueError("Original M0/MZ scores and participant/version identities must agree exactly")
    for number, (fit, (model, direction, train, held)) in enumerate(zip(fits, expected)):
        if (fit.get("model") != model or fit.get("rival") != "fixed" or fit.get("training_runs") != list(train)
                or fit.get("scoring_runs") != list(held) or fit.get("participants") != sorted(baseline)
                or type(fit.get("converged")) is not bool):
            raise ValueError("Returned technical fit identity or ordered direction changed")
        duration(fit.get("elapsed_seconds"))
        if number < len(scored):
            fixed.validate_fit(fit, model, train, held, spec, baseline)
    for row, (model, direction, train, held) in zip(scored, expected):
        if (row.get("model") != model or row.get("direction") != direction
                or row.get("training_runs") != list(train) or row.get("scoring_runs") != list(held)
                or row.get("heldout_trials_per_participant") != 128 or row.get("per_participant_denominator") != 256
                or not isinstance(row.get("participant_contributions"), dict)
                or set(row["participant_contributions"]) != set(baseline)
                or any(not twice.finite(value) for value in row["participant_contributions"].values())):
            raise ValueError("Scored halves require exact128/256 contributions for all108 original people")
    admitted = {}
    for model in completed:
        halves = [row for row in scored if row["model"] == model]
        if len(halves) != 2 or any(panel[p][model] != halves[0]["participant_contributions"][p]
                + halves[1]["participant_contributions"][p] for p in baseline):
            raise ValueError("A completed model must equal both validated held-out contributions")
        admitted[model] = {person: panel[person][model] for person in baseline}
    if not technical:
        if (len(fits) != len(expected) or len(scored) != len(expected) or completed != list(models)
                or report.get("parallel_workers_cleaned_up") is not True):
            raise ValueError("Completed reports require every fit/score and successful worker cleanup")
        if is_pilot:
            calculated = pilot.menu_diagnostics(panel, SimpleNamespace(menu=fixed))
            if any(report.get(field) != value for field, value in calculated.items()):
                raise ValueError("Completed pilot diagnostics differ from its validated score panel")
    # A terminal technical indexed component is unknown even if its final
    # publication failed after retaining a valid pair. Prefix salvage is the
    # explicit exception for the original multi-model index0 pilot only.
    return ({} if technical and not is_pilot else admitted), metadata


def unknown_rows():
    return [{"replicate_index": index, "seed": twice.levels.level_seed("twice", index), "status": "unknown",
             "reason": "not_selected_as_terminal_by_coordinator", "components": {}} for index in range(200)]


def summarize(rows, spec, indices, declarations):
    complete = [row for row in rows if row["status"] == "complete_menu"]
    maxima = Counter("+".join(row["maximal_eligible_models_diagnostic"]) for row in complete)
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "implementation": spec,
        "generating_model": "MZ", "effect_level": "twice", "SESOI_level": 2., "generating_z": .40,
        "fixed_cohort_count": 200, "selected_terminal_indices": indices,
        "terminal_component_declarations": [{"model": model, "array_job_id": array, "indices": selected}
            for model, array, selected in declarations], "complete_menus": len(complete),
        "unknown_menus": 200 - len(complete), "unknown_indices": [row["replicate_index"] for row in rows if row["status"] == "unknown"],
        "MZ_only_maximal_count": sum(row["maximal_eligible_models_diagnostic"] == ["MZ"] for row in complete),
        "incomparable_maximal_count": sum(row["multiple_incomparable_eligible_models"] for row in complete),
        "maximal_set_counts_descriptive": dict(sorted(maxima.items())), "cohorts": rows,
        **{field: "not_evaluated" for field in CLAIMS}, "full_recovery_pass": None, "episode_terminal": False,
        "new_simulation": False, "new_model_fits": False, "source_events_read": False, "saved_scores_rescored": False,
        "automatic_retry": False, "wave_selected": False,
        "interpretation": "Fixed200 descriptive menus. Nine original nested contrasts and partial-order maximal sets only; no arbitration, family/rate/interval or full-gate verdict. Missing and technical components remain unknown; validated completed pilot model prefixes may be reused.",
        "cost_accounting": "Each baseline and selected attempt is retained once per cohort; costs are references only, never summed or recharged.",
        "process_state_boundary": "Coordinator-established terminal selections only; report fields never replace scheduler evidence."}


def pool(indices, declarations, spec, *, check=lambda: None, rows=None):
    chosen = selection(indices, declarations, spec)
    rows = unknown_rows() if rows is None else rows
    versions = None
    for index in sorted(indices):
        row = rows[index]
        check()
        path = baseline_path(index)
        row.update(baseline_report=str(path), reason="selected_terminal_baseline_missing")
        try:
            base = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        check()
        twice.validate_identity(base, index, spec)
        row["baseline_attempt"] = terminal(base)
        if base.get("status") == "technical_recovery_failure":
            row.update(reason="technical_baseline_failure", baseline_technical=twice.technical_metadata(base))
            continue
        twice.completed_diagnostic(base, spec)
        scores = deepcopy(fixed.score_panel(base, ("M0", "MZ")))
        current_versions = {person: score["version"] for person, score in scores.items()}
        if versions is not None and current_versions != versions:
            raise ValueError("Participant/version identities changed across the fixed twice-cell cohort set")
        versions = current_versions
        row.update(reason="incomplete_component_menu", attempts=[])
        inputs = [(PILOT_REPORT, pilot.RIVALS, True)] if index == 0 else [
            (component_path(index, model, chosen[(index, model)]), (model,), False)
            for model in pilot.RIVALS if (index, model) in chosen]
        row["components"] = {model: {"status": "unknown", "reason": "not_selected_as_terminal_by_coordinator"} for model in pilot.RIVALS}
        for report_path, models, is_pilot in inputs:
            check()
            for model in models:
                row["components"][model].update(report=str(report_path), reason="selected_terminal_report_missing")
            try:
                report = json.loads(report_path.read_text())
            except FileNotFoundError:
                continue
            check()
            admitted, evidence = validate_rivals(report, base, index, spec, models, is_pilot=is_pilot)
            evidence["report"] = str(report_path)
            row["attempts"].append(evidence)
            for model in models:
                component = row["components"][model]
                if model in admitted:
                    component.update(status="completed_pair", from_technical_attempt=report["status"].startswith("technical_"))
                    component.pop("reason", None)
                    for person in scores:
                        scores[person][model] = admitted[model][person]
                else:
                    component["reason"] = "technical_component_unknown"
        if all(component["status"] == "completed_pair" for component in row["components"].values()):
            check()
            row.update(status="complete_menu", participant_scores=scores,
                       **pilot.menu_diagnostics(scores, SimpleNamespace(menu=fixed)))
            row.pop("reason", None)
    check()
    return summarize(rows, spec, sorted(indices), declarations)


def write_report(output, report):
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-indices", type=twice.terminal_indices, required=True)
    parser.add_argument("--component-array", type=component_array, action="append", default=[])
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the original implementation path")
    if not 1 <= args.max_seconds <= MAX_SECONDS or any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise ValueError("Require a1..240-second whole-process cap and all six numerical threads one")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError("Preserve prior outputs; select a fresh summary")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    rows, spec = unknown_rows(), None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Whole-process twice-menu collection deadline exceeded")

    def interrupt(signum, frame):
        raise RuntimeError(f"Twice-menu collection interrupted by signal {signum}")

    def publish(result):
        result.update(recorded_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                      internal_seconds=args.max_seconds, job_id=os.environ.get("SLURM_JOB_ID"))
        write_report(args.output, result)

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupt)
    signal.alarm(args.max_seconds)
    succeeded, failure = False, None
    try:
        check()
        spec = json.loads(IMPLEMENTATION.read_text())
        result = pool(args.terminal_indices, args.component_array, spec, check=check, rows=rows)
        result["status"] = "terminal_selection_collected"
        check()
        publish(result)
        check()
        succeeded = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if succeeded:
            try:
                check()
            except BaseException as error:
                succeeded, failure = False, f"{type(error).__name__}: {error}"
        if not succeeded:
            result = summarize(rows, spec, args.terminal_indices, args.component_array)
            result.update(status="technical_menu_collection_failure", error=failure)
            publish(result)
    print(json.dumps({field: result[field] for field in ("status", "complete_menus", "unknown_menus",
        "MZ_only_maximal_count", "incomparable_maximal_count", "error") if field in result}), flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
