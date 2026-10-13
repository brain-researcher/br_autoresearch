"""One explicit numerical continuation of EP07's fixed four-case suite.

The original generation, oracles, diagnostics and recovery gates are imported
unchanged. Reuse the completed stable/M1/history cell and its two member fits.
The failed stable/M1/today/starting_09 fit has no checkpoint: its one retry
starts from zero. Only starting_09's eight today-only task executions receive
a 100000 iteration ceiling; its statistical recipe and KKT tolerance stay
fixed. The seven unattempted executions start directly with that ceiling.

There are 30 new fits, 32 unique successful member predictions, and 33 total
attempts including the old failed fit. No completed cell is refit or rescored.
Any further numerical or scientific failure stops without another retry.
Charge the prior 13.77209186553955 active seconds conservatively as 14, leaving
at most 586 active seconds. This module never reads empirical neural data.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from numbers import Real
import os
from pathlib import Path
import resource
import socket
import time

import known_answer_cases as fixed


PRIOR_DIRECTORY = "outputs/runs/alternative_explanations/attempt001"
PRIOR_REPORT = PRIOR_DIRECTORY + "/known_answer_report.json"
SELECTION = PRIOR_DIRECTORY + "/selected_pair.json"
REPAIR_DIRECTORY = "outputs/runs/alternative_explanations/qualification_repair001"
REPAIR_REPORT = REPAIR_DIRECTORY + "/known_answer_report.json"
EXCEPTION_ID = "known_answer_single_numerical_repair_20261011"
PRIOR_WALL_SECONDS = 13.77209186553955
PRIOR_CPU_SECONDS = 12.336662184
PRIOR_WALL_CHARGED_SECONDS = 14.
CONTINUATION_LIMIT_SECONDS = 586.
SOLVER_ITERATION_CEILING = 100000
REUSED_CELL = ("stable", "M1", "history_assisted")
FAILED_KEY = ("stable", "M1", "today_only", "starting_09")
PRIOR_ERROR = ("elastic net exhausted its iteration limit before the specified KKT tolerance: "
    "iterations=1000, residual_kkt_error=0.0026085344030212449, "
    "gram_kkt_error=0.0026085344030515817, tolerance=1.0000000000000001e-05")


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _keys(finalists):
    by_stream = {value.stream: value for value in finalists}
    return tuple((case, region, stream, member.candidate_id)
        for case in fixed.CASES for region in ("M1", "PMd") for stream in fixed.STREAMS
        for member in by_stream[stream].members)


def _cell_keys():
    return tuple((case, region, stream) for case in fixed.CASES
                 for region in ("M1", "PMd") for stream in fixed.STREAMS)


def _admit_prior(prior, finalists):
    feasibility = fixed.qualify_finalists(finalists)
    ids = {value.stream: value.candidate_id for value in finalists}
    if ids != {"history_assisted": "adaptive_07", "today_only": "adaptive_03"}:
        raise ValueError("repair binds only the unchanged already selected finalists")
    if (prior.get("state") != "unresolved_qualification" or prior.get("stage") != "member_fit"
            or prior.get("known_answer_recovery_complete") is not False
            or prior.get("held_out_opened") is not False
            or prior.get("scientific_checks_complete") is not False
            or prior.get("evidence_stage") != "generated_known_answer_only"
            or prior.get("selection") != SELECTION or prior.get("finalists") != ids
            or prior.get("animal") != "Mihili" or prior.get("target_day") != "20140304"
            or prior.get("cases") != list(fixed.CASES) or prior.get("seed") != fixed.SEED
            or prior.get("internal_limit_seconds") != 600
            or prior.get("allocated_cpus") != 1 or prior.get("numerical_threads") != 1
            or prior.get("attempted_member_fits") != 3
            or prior.get("completed_member_predictions") != 2
            or prior.get("wall_seconds") != PRIOR_WALL_SECONDS
            or prior.get("process_cpu_seconds") != PRIOR_CPU_SECONDS
            or tuple(prior.get(name) for name in ("case", "region", "stream", "member_id")) != FAILED_KEY
            or prior.get("error_type") != "RegressionNotConverged" or prior.get("error") != PRIOR_ERROR
            or prior.get("feasibility") != _json(feasibility)):
        raise ValueError("prior report is not the exact admitted numerical failure")
    results = prior.get("results", [])
    if (len(results) != 1 or tuple(results[0].get(name) for name in ("case", "region", "stream")) != REUSED_CELL
            or results[0].get("finalist_id") != ids[REUSED_CELL[2]] or results[0].get("passed") is not True):
        raise ValueError("exactly the original completed passing stable/M1/history cell is reusable")
    _validate_result(results[0], finalists[0])
    if len(_keys(finalists)) != 32:
        raise ValueError("repair requires the original32 unique member tasks")
    return feasibility


def _validate_result(result, finalist):
    if set(result) != set(fixed.RecoveryResult.__dataclass_fields__):
        raise ValueError("complete original recovery-result fields required")
    if (result["finalist_id"] != finalist.candidate_id or result["stream"] != finalist.stream
            or result["case"] not in fixed.CASES or result["region"] not in ("M1", "PMd")
            or result["passed"] is not True):
        raise ValueError("every fixed recovery cell must pass for its unchanged finalist")
    names = ("oracle_h_amplitude", "oracle_beta", "prediction_beta", "nrmse", "slope_error", "slope_limit")
    if any(isinstance(result[name], bool) or not isinstance(result[name], Real)
           or not math.isfinite(float(result[name])) for name in names):
        raise ValueError("recovery statistics must be finite numbers")
    oracles = tuple(fixed.member_oracle(member, result["case"], stream=finalist.stream)
                    for member in finalist.members)
    amplitude = math.fsum(member.weight * value[0] for member, value in zip(finalist.members, oracles))
    beta = math.fsum(member.weight * value[1] for member, value in zip(finalist.members, oracles))
    if (result["oracle_h_amplitude"] != amplitude or result["oracle_beta"] != beta
            or result["slope_error"] != abs(result["prediction_beta"] - beta)
            or result["slope_limit"] != .20 * max(abs(beta), 1.)
            or not 0 <= result["nrmse"] <= fixed.NRMSE_LIMIT
            or not 0 <= result["slope_error"] <= result["slope_limit"]):
        raise ValueError("unchanged oracle, NRMSE and slope criteria must hold")


def _member(finalists, key):
    return next(member for value in finalists if value.stream == key[2]
                for member in value.members if member.candidate_id == key[3])


def _execution_recipe(member, stream):
    if stream == "today_only" and member.candidate_id == "starting_09":
        if (member.recipe.regression.family != "elastic_net"
                or member.recipe.regression.max_iterations != 1000
                or member.recipe.regression.tolerance != 1e-5):
            raise ValueError("solver override requires the unchanged admitted elastic-net declaration")
        return replace(member.recipe, regression=replace(member.recipe.regression,
                                                        max_iterations=SOLVER_ITERATION_CEILING))
    return member.recipe


def _attempt(number, origin, limit, *, status="completed_prediction", error=None):
    value = {"attempt": number, "origin": origin, "max_iterations": limit, "status": status}
    if error is not None:
        value.update(error_type="RegressionNotConverged", error=error)
    return value


def _initial_ledger(finalists):
    ledger = []
    for key in _keys(finalists):
        member = _member(finalists, key)
        reused = key[:3] == REUSED_CELL
        attempts = []
        if reused:
            attempts = [_attempt(1, PRIOR_REPORT, member.recipe.regression.max_iterations)]
        elif key == FAILED_KEY:
            attempts = [_attempt(1, PRIOR_REPORT, 1000, status="numerical_failure", error=PRIOR_ERROR)]
        ledger.append({"key": list(key), "reused_completed_prediction": reused,
                       "completed_prediction": reused, "attempts": attempts})
    return ledger


def _counts(ledger):
    return (sum(len(record["attempts"]) for record in ledger),
            sum(record["completed_prediction"] for record in ledger))


def _base_report(prior, finalists, feasibility):
    return {"state": "running", "evidence_stage": "generated_known_answer_only",
        "held_out_opened": False, "scientific_checks_complete": False,
        "known_answer_recovery_complete": False, "exception_id": EXCEPTION_ID,
        "selection": SELECTION, "prior_report": PRIOR_REPORT,
        "cases": list(fixed.CASES), "seed": fixed.SEED,
        "finalists": {value.stream: value.candidate_id for value in finalists},
        "finalist_specs": _json({value.stream: asdict(value) for value in finalists}),
        "feasibility": _json(feasibility), "failed_task_retried_once": list(FAILED_KEY),
        "reused_completed_cells": [list(REUSED_CELL)],
        "solver_override": {"member_id": "starting_09", "stream": "today_only",
            "field": "regression.max_iterations", "from": 1000, "to": SOLVER_ITERATION_CEILING,
            "task_count": 8, "all_other_recipe_fields_unchanged": True},
        "prior_wall_seconds": prior["wall_seconds"], "prior_process_cpu_seconds": prior["process_cpu_seconds"],
        "prior_wall_charged_seconds": PRIOR_WALL_CHARGED_SECONDS,
        "continuation_limit_seconds": CONTINUATION_LIMIT_SECONDS, "total_limit_seconds": 600,
        "results": _json(prior["results"]), "member_task_ledger": _initial_ledger(finalists),
        "attempted_member_fits": 3, "completed_member_predictions": 2,
        "new_member_fit_attempts": 0, "unique_member_tasks": 32}


def run_remaining_cases(finalists, prior_report, *, check, progress=None):
    """Complete only uncompleted cells; any exception propagates without retry."""
    from native_trials import _snapshot
    from primary_model import fit_primary_model, predict_primary_development
    from weighted_primary_model import fit_weighted_primary_model, predict_weighted_primary_development
    fixed._check(check)
    feasibility = _admit_prior(prior_report, finalists)
    report = _base_report(prior_report, finalists, feasibility)
    ledger = {tuple(record["key"]): record for record in report["member_task_ledger"]}
    by_stream = {value.stream: value for value in finalists}
    def emit(stage):
        report["stage"] = stage
        report["attempted_member_fits"], report["completed_member_predictions"] = _counts(report["member_task_ledger"])
        report["new_member_fit_attempts"] = report["attempted_member_fits"] - 3
        if progress is not None:
            progress(_json(report))
    emit("prior_completed_cell_reused")
    for case in fixed.CASES:
        fixed._check(check)
        generated = fixed.generate_case(case, check=check)
        for cell in generated.cells:
            if (case, cell.region, cell.stream) == REUSED_CELL:
                continue
            finalist, combined, reference = by_stream[cell.stream], None, None
            for member in finalist.members:
                fixed._check(check)
                key = (case, cell.region, cell.stream, member.candidate_id)
                record = ledger[key]
                if record["completed_prediction"] or len(record["attempts"]) != int(key == FAILED_KEY):
                    raise ValueError("no additional retry or completed member fit is permitted")
                recipe = _execution_recipe(member, cell.stream)
                attempt = _attempt(len(record["attempts"]) + 1, REPAIR_REPORT,
                                   recipe.regression.max_iterations, status="running")
                record["attempts"].append(attempt)
                report.update(case=case, region=cell.region, stream=cell.stream, member_id=member.candidate_id)
                emit("member_fit")
                fitted = prediction = None
                try:
                    if member.weight_recipe is None:
                        fitted = fit_primary_model(cell.fit_days, recipe=recipe, check=check)
                        prediction = predict_primary_development(fitted, cell.development, check=check)
                    else:
                        fitted = fit_weighted_primary_model(cell.fit_days, recipe=recipe,
                            weight_recipe=member.weight_recipe, check=check)
                        prediction = predict_weighted_primary_development(fitted, cell.development, check=check)
                    fixed._fitted_diagnostics(fitted, member, cell)
                    fixed._prediction_values(cell, prediction)
                    reference = prediction
                    weighted = member.weight * prediction.values
                    combined = weighted.copy() if combined is None else combined + weighted
                    attempt["status"] = "completed_prediction"
                    record["completed_prediction"] = True
                    emit("member_prediction_completed")
                except Exception as error:
                    attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
                    report.update(state="unresolved_qualification", error_type=type(error).__name__, error=str(error))
                    emit("member_failed_no_retry")
                    raise
                finally:
                    fitted = prediction = None
            fixed._check(check)
            result = fixed.evaluate_prediction(cell, replace(reference, values=_snapshot(combined)),
                                               case=case, finalist=finalist)
            report["results"].append(asdict(result))
            if not result.passed:
                report["state"] = "scientific_recovery_failed"
                emit("scientific_failure_no_retry")
                return report
            emit("cell_completed")
        del generated
    report.update(state="all_four_cases_passed", known_answer_recovery_complete=True)
    emit("finished")
    return report


def validate_completed_report(report, prior_report, finalists):
    """Validate the exact documented continuation for primary eligibility.

The caller reads the fixed immutable original report and selected finalists;
this function does not choose a report path or authorize outcome access.
"""
    feasibility = _admit_prior(prior_report, finalists)
    expected = _base_report(prior_report, finalists, feasibility)
    stable_fields = ("evidence_stage", "held_out_opened", "scientific_checks_complete", "exception_id",
        "selection", "prior_report", "cases", "seed", "finalists", "finalist_specs", "feasibility",
        "failed_task_retried_once", "reused_completed_cells", "solver_override", "prior_wall_seconds",
        "prior_process_cpu_seconds", "prior_wall_charged_seconds", "continuation_limit_seconds",
        "total_limit_seconds", "unique_member_tasks")
    if any(report.get(name) != expected[name] for name in stable_fields):
        raise ValueError("continuation declarations or exact immutable prior binding changed")
    if (report.get("state") != "all_four_cases_passed" or report.get("stage") != "finished"
            or report.get("known_answer_recovery_complete") is not True
            or report.get("attempted_member_fits") != 33
            or report.get("completed_member_predictions") != 32
            or report.get("new_member_fit_attempts") != 30):
        raise ValueError("continuation must finish exactly33 attempts and32 unique member predictions")
    expected_ledger = expected["member_task_ledger"]
    for record in expected_ledger:
        key = tuple(record["key"])
        if not record["completed_prediction"]:
            recipe = _execution_recipe(_member(finalists, key), key[2])
            record["attempts"].append(_attempt(len(record["attempts"]) + 1, REPAIR_REPORT,
                                               recipe.regression.max_iterations))
            record["completed_prediction"] = True
    if report.get("member_task_ledger") != expected_ledger:
        raise ValueError("exact complete member ledger required; no extra retry, repeat or omitted task")
    results = report.get("results", [])
    if (len(results) != 16 or tuple(tuple(value.get(name) for name in ("case", "region", "stream"))
                                   for value in results) != _cell_keys()
            or results[0] != prior_report["results"][0]):
        raise ValueError("all16 fixed cells required with the original completed cell reused exactly")
    by_stream = {value.stream: value for value in finalists}
    for result in results:
        _validate_result(result, by_stream[result["stream"]])
    timing = ("continuation_wall_seconds", "continuation_process_cpu_seconds",
              "cumulative_wall_seconds", "cumulative_wall_charged_seconds", "cumulative_process_cpu_seconds")
    if any(isinstance(report.get(name), bool) or not isinstance(report.get(name), Real)
           or not math.isfinite(float(report[name])) or report[name] < 0 for name in timing):
        raise ValueError("finite nonnegative actual and charged continuation times required")
    if (report["continuation_wall_seconds"] > CONTINUATION_LIMIT_SECONDS
            or report["cumulative_wall_seconds"] != PRIOR_WALL_SECONDS + report["continuation_wall_seconds"]
            or report["cumulative_wall_charged_seconds"] != PRIOR_WALL_CHARGED_SECONDS + report["continuation_wall_seconds"]
            or report["cumulative_wall_charged_seconds"] > 600
            or report["cumulative_process_cpu_seconds"] != PRIOR_CPU_SECONDS + report["continuation_process_cpu_seconds"]):
        raise ValueError("prior time cannot be reset; active suite wall charge must remain within600 seconds")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    started_wall, started_cpu = time.time(), time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    episode = Path(__file__).resolve().parents[2]
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in socket.gethostname()
            or os.environ.get("SLURM_CPUS_PER_TASK") not in ("1", "2")
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"
            or any(os.environ.get(name) != "1" for name in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
            or not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= started_wall):
        raise SystemExit("continuation requires the selected bounded compute allocation")
    if (args.selection.resolve() != episode / SELECTION or args.prior_report.resolve() != episode / PRIOR_REPORT
            or args.out.resolve() != episode / REPAIR_REPORT or not args.out.parent.is_dir()
            or any(path.stat().st_size > 1024**2 for path in (args.selection, args.prior_report))):
        raise SystemExit("continuation input/output differ from the explicitly selected attempts")
    selection, prior = json.loads(args.selection.read_text()), json.loads(args.prior_report.read_text())
    if (selection.get("evidence_stage") != "development_only" or selection.get("held_out_opened") is not False
            or selection.get("scientific_checks_complete") is not False
            or set(selection.get("finalists", {})) != set(fixed.STREAMS)):
        raise SystemExit("selection must remain the unopened development-only finalist pair")
    deadline = min(args.deadline_epoch, started_wall + CONTINUATION_LIMIT_SECONDS)
    def check():
        if time.time() >= deadline:
            raise TimeoutError("remaining586-second known-answer continuation deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("unchanged known-answer RSS limit exceeded")
    status = {"state": "running", "stage": "continuation_admission", "started_utc": _utc(),
              "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "allocated_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]), "numerical_threads": 1,
              "deadline_epoch": deadline, "known_answer_recovery_complete": False}
    def timing():
        wall, cpu = time.time() - started_wall, time.process_time() - started_cpu
        status.update(updated_utc=_utc(), continuation_wall_seconds=wall,
            continuation_process_cpu_seconds=cpu, cumulative_wall_seconds=PRIOR_WALL_SECONDS + wall,
            cumulative_wall_charged_seconds=PRIOR_WALL_CHARGED_SECONDS + wall,
            cumulative_process_cpu_seconds=PRIOR_CPU_SECONDS + cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    with args.out.open("x") as handle:
        def save():
            timing()
            target = Path(handle.name)
            temporary = target.with_name(target.name + ".tmp")
            with temporary.open("x") as output:
                json.dump(status, output, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        save()
        try:
            check()
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from finalist_controls import resolve_finalist
            finalists = tuple(resolve_finalist(selection["finalists"][stream]["candidate_id"], stream=stream)
                               for stream in fixed.STREAMS)
            for finalist in finalists:
                if selection["finalists"][finalist.stream].get("spec") != _json(asdict(finalist)):
                    raise ValueError("selected original declaration differs from its fixed recipe")
            def progress(value):
                status.update(value)
                save()
            status.update(run_remaining_cases(finalists, prior, check=check, progress=progress))
            timing()
            if status["state"] == "all_four_cases_passed":
                check()
                validate_completed_report(status, prior, finalists)
            save()
        except Exception as error:
            status.update(state="unresolved_qualification", error_type=type(error).__name__,
                          error=str(error), known_answer_recovery_complete=False)
            save()
            return 1
    return 0 if status["state"] == "all_four_cases_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
