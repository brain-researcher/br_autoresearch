#!/usr/bin/env python3
"""Bounded, role-separated EP07 finalist selection, control fitting and scoring.

Each invocation performs exactly one mode. The coordinator owns Slurm, serial
control order and stopping. Candidate fitting receives no scorer path; scoring
receives no candidate path and begins outcome contact only after BOTH complete
control panels and their declared original finalists are validated. Selection
uses the independent global stream incumbents after completed adaptive search;
it never substitutes a runner-up or requires known-answer results to select.

All ten controls remain mandatory. Invalid fits/publications, unavailable scores
or partial contacts stop continuation. Score changes are sensitivity evidence,
not an invented degradation veto. Failed artifacts and the original search clock
are preserved. No held-out path, automatic retry, scheduler call or launch exists.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time


STREAMS = ("history_assisted", "today_only")


def utc():
    return datetime.now(timezone.utc).isoformat()


def _json_array(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported operational metadata type: {type(value).__name__}")


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False, default=_json_array))


def _record(handle, value):
    """Atomically update this process's exclusively claimed status path.

    A failed post-contact update leaves the previous complete contact snapshot
    readable. A failed temporary file is preserved; it is never overwritten or
    treated as a successful status. The original x-open handle identifies the
    path claimed by this process, even after its first atomic replacement.
    """
    target = Path(handle.name)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False, default=_json_array)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)


def _clock(base, bind_deadline, check):
    check()
    clock = json.loads((base / "search_clock.json").read_text())
    start, limit = clock.get("started_epoch"), clock.get("wall_clock_limit_seconds")
    if (isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start)
            or isinstance(limit, bool) or not isinstance(limit, (int, float))
            or limit != 345600 or not isinstance(clock.get("started_utc"), str)
            or not clock["started_utc"]):
        raise ValueError("the original finite 96-hour search clock is required")
    bind_deadline(start + limit)
    check()
    return clock


def _admission(base, last_index, bind_deadline, check):
    from adaptive_progress import assess_adaptive_progress
    from run_adaptive_models import _prior_feedback
    if type(last_index) is not int or last_index not in range(14):
        raise ValueError("last_adaptive_index must identify an evaluated adaptive pair")
    clock = _clock(base, bind_deadline, check)
    starts, previous, members = _prior_feedback(base, last_index + 1, check)
    progress = assess_adaptive_progress(starts, previous)
    if (not progress.starting_complete or not progress.adaptive_stage_complete
            or progress.incomplete_contact_ids or any(value is None for value in progress.incumbents)):
        raise ValueError("finalist selection requires complete starts, completed adaptive stage and closed contacts")
    return clock, progress, members


def _finalists(progress):
    from finalist_controls import resolve_finalist
    return {stream: resolve_finalist(leader.candidate_id, stream=stream)
            for stream, leader in zip(STREAMS, progress.incumbents)}


def _select(args, base, bind_deadline, check):
    clock, progress, _ = _admission(base, args.last_index, bind_deadline, check)
    specs = _finalists(progress)
    selected = {"last_adaptive_index": args.last_index, "search_clock": clock,
        "progress": asdict(progress), "finalists": {
            stream: {"candidate_id": spec.candidate_id, "spec": asdict(spec)}
            for stream, spec in specs.items()}, "selection_completed_utc": utc(),
        "evidence_stage": "development_only", "held_out_opened": False,
        "scientific_checks_complete": False}
    check()
    with (args.out / "selected_pair.json").open("x") as handle:
        _record(handle, selected)
    print(json.dumps({"mode": "select", "finalists": {stream: spec.candidate_id
          for stream, spec in specs.items()}, "held_out_opened": False}), flush=True)
    return 0


def _selected(args, base, bind_deadline, check):
    check()
    selected = json.loads((args.out / "selected_pair.json").read_text())
    if (selected.get("evidence_stage") != "development_only"
            or selected.get("held_out_opened") is not False
            or selected.get("scientific_checks_complete") is not False
            or not selected.get("selection_completed_utc")):
        raise ValueError("a completed development-only selection is required")
    clock, progress, members = _admission(base, selected.get("last_adaptive_index"), bind_deadline, check)
    specs = _finalists(progress)
    expected = {stream: {"candidate_id": spec.candidate_id, "spec": asdict(spec)}
                for stream, spec in specs.items()}
    if (selected.get("search_clock") != clock or selected.get("progress") != _plain(asdict(progress))
            or selected.get("finalists") != _plain(expected)):
        raise ValueError("selected finalists/recipes/progress/clock differ from the admitted search")
    return selected, specs, progress, members


def _ids(specs):
    return {stream: specs[stream].candidate_id for stream in STREAMS}


def _previous_controls(out, number, specs, check):
    from ensemble_predictions import _evaluated
    for prior in range(1, number):
        check()
        name = f"control_{prior:02d}"
        report = json.loads((out / f"{name}_feedback.json").read_text())
        if (report.get("control") != prior or report.get("evidence_stage") != "development_only"
                or report.get("held_out_opened") is not False or not report.get("ended_utc")
                or report.get("complete") is not True
                or report.get("intervention_integrity_complete") is not True
                or report.get("finalists") != _ids(specs)):
            raise ValueError("every preceding paired control must be closed, valid and fully available")
        for stream in STREAMS:
            _evaluated(report.get("pair"), name, stream)


def _admitted_panels(specs, progress, records, check, *, include_members):
    from candidate_predictions import load_candidate_predictions
    from ensemble_predictions import _evaluated
    valid = {value.candidate_id for value in progress.starting_records + progress.successor_records
             if value.status == "valid_pair"}
    panels = {}
    for stream, spec in specs.items():
        names = {spec.candidate_id}
        if include_members:
            names.update(member.candidate_id for member in spec.members)
        for name in sorted(names):
            check()
            if name not in valid or name not in records:
                raise ValueError("every original finalist/member needs a complete available paired evaluation")
            record, directory = records[name]
            for paired_stream in STREAMS:
                _evaluated(record, name, paired_stream)
            panels[(stream, name)] = load_candidate_predictions(directory / f"{name}__{stream}.h5",
                expected_candidate_id=name, expected_stream=stream, check=check)
    return panels


def _cells(panel):
    return {(value.animal, value.target_day, value.region): (value, baseline)
            for value, baseline in zip(panel.predictions, panel.calibration_means)}


def _integrity(plan, panel, original, check):
    import numpy as np
    from ensemble_predictions import _same_cell
    original_cells = _cells(original)
    wholly_reused = all(member.action == "reuse" for member in plan.members)
    template = all(member.action == "template" for member in plan.members)
    for key, pair in _cells(panel).items():
        check()
        _same_cell(pair, original_cells[key])
        if wholly_reused and not np.array_equal(pair[0].values, original_cells[key][0].values):
            raise ValueError("a wholly reused finalist must exactly match its original published predictions")
        if template and not np.array_equal(pair[0].values, pair[1].values):
            raise ValueError("template-only control must exactly equal the calibration comparator")
    return {"complete": True, "wholly_reused_exact": wholly_reused,
            "template_comparator_exact": template,
            "score_changes": "sensitivity_only_no_degradation_veto"}


def _fit(args, specs, progress, records, check):
    from candidate_predictions import CandidatePredictionWriter
    from finalist_controls import plan_control, run_member_cell, compose_control_panel
    from fit_mismatch import derange_fit_lfp
    from native_trials import NEW, load_native_trials
    from view_access import CandidateViews
    name = f"control_{args.control:02d}"
    if ((args.out / f"{name}_feedback.json").exists()
            or any((args.out / f"{name}__{stream}.h5").exists() for stream in STREAMS)):
        raise FileExistsError("this control already has artifacts; preserve them without retry")
    plans = {stream: plan_control(specs[stream], args.control) for stream in STREAMS}
    if any(member.member.recipe.regression.family == "shallow_network"
           for plan in plans.values() for member in plan.members):
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    status = {"control": args.control, "candidate_id": name, "finalists": _ids(specs),
        "plans": {stream: asdict(plan) for stream, plan in plans.items()},
        "evidence_stage": "development_only", "held_out_opened": False,
        "outcome_contact": False, "state": "running", "complete": False,
        "started_utc": utc(), "job_id": os.environ["SLURM_JOB_ID"], "settings": []}
    with (args.out / f"{name}_worker.json").open("x") as handle:
        _record(handle, status)
        try:
            originals = _admitted_panels(specs, progress, records, check, include_members=True)
            original_cells = {key: _cells(value) for key, value in originals.items()}
            results = {stream: {member.member.candidate_id: [] for member in plan.members}
                       for stream, plan in plans.items()}
            views = CandidateViews(args.candidate_directory)
            for animal in ("Mihili", "Chewie-L"):
                for target in NEW[animal]:
                    check()
                    fits, developments = {}, {}
                    for region in ("M1", "PMd"):
                        for stream in STREAMS:
                            scope = dict(animal=animal, target_day=target, region=region, stream=stream)
                            fits[(region, stream)] = load_native_trials(views, view_role="fit", check=check, **scope)
                            developments[(region, stream)] = load_native_trials(
                                views, view_role="development_features", check=check, **scope)[0]
                    packs = tuple(day for days in fits.values() for day in days)
                    mismatch = (derange_fit_lfp(packs, scope="earlier" if args.control == 2 else "calibration",
                                                check=check) if args.control in (2, 3) else None)
                    for region in ("M1", "PMd"):
                        for stream in STREAMS:
                            for member in plans[stream].members:
                                check()
                                member_id = member.member.candidate_id
                                status["current_setting"] = [animal, target, region, stream, member_id]
                                _record(handle, status)
                                original, baseline = original_cells[(stream, member_id)][(animal, target, region)]
                                started = time.monotonic()
                                result = run_member_cell(plans[stream], member_id,
                                    fit_days=fits[(region, stream)], development=developments[(region, stream)],
                                    original_prediction=original, calibration_mean=baseline,
                                    mismatch=mismatch, check=check)
                                results[stream][member_id].append(result)
                                status["settings"].append({"setting": status["current_setting"],
                                    "seconds": time.monotonic() - started, "diagnostics": asdict(result.diagnostics)})
                                _record(handle, status)
                    del fits, developments, packs, mismatch
            panels, integrity = {}, {}
            for stream in STREAMS:
                plan = plans[stream]
                panels[stream] = compose_control_panel(plan, tuple(tuple(results[stream][member.member.candidate_id])
                    for member in plan.members), check=check)
                integrity[stream] = _integrity(plan, panels[stream], originals[(stream, specs[stream].candidate_id)], check)
            # Finish BOTH complete mathematical panels before publishing either.
            status.update(state="publishing", integrity=integrity)
            _record(handle, status)
            for stream in STREAMS:
                with CandidatePredictionWriter(args.out / f"{name}__{stream}.h5",
                        candidate_id=name, stream=stream, check=check) as writer:
                    for prediction, baseline in zip(panels[stream].predictions, panels[stream].calibration_means):
                        writer.write(prediction, baseline)
            status.update(state="paired_predictions_complete_unscored", complete=True)
        except BaseException as error:
            status.update(state="failed_incomplete", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            status["ended_utc"] = utc()
            _record(handle, status)
    print(json.dumps({"mode": "fit", "control": args.control, "state": status["state"]}), flush=True)
    return 0


def _score(args, specs, progress, records, check):
    from candidate_predictions import load_candidate_predictions
    from ensemble_predictions import _same_cell
    from finalist_controls import plan_control
    from development_score import score_development_panel
    name = f"control_{args.control:02d}"
    worker = json.loads((args.out / f"{name}_worker.json").read_text())
    if (worker.get("control") != args.control or worker.get("candidate_id") != name
            or worker.get("finalists") != _ids(specs) or worker.get("complete") is not True
            or worker.get("state") != "paired_predictions_complete_unscored"
            or worker.get("outcome_contact") is not False or not worker.get("ended_utc")
            or worker.get("evidence_stage") != "development_only" or worker.get("held_out_opened") is not False):
        raise ValueError("the exact completed paired control fit is required before scoring")
    pair = {"candidate_id": name, "outcome_contact": False,
            "state": "paired_prediction_incomplete_unscored", "streams": {}}
    report = {"control": args.control, "finalists": _ids(specs), "pair": pair,
        "evidence_stage": "development_only", "held_out_opened": False,
        "complete": False, "intervention_integrity_complete": False,
        "scientific_checks_complete": False, "score_changes": "sensitivity_only_no_degradation_veto",
        "started_utc": utc(), "job_id": os.environ["SLURM_JOB_ID"]}
    with (args.out / f"{name}_feedback.json").open("x") as handle:
        _record(handle, report)
        try:
            originals = _admitted_panels(specs, progress, records, check, include_members=False)
            panels = {stream: load_candidate_predictions(args.out / f"{name}__{stream}.h5",
                expected_candidate_id=name, expected_stream=stream, check=check) for stream in STREAMS}
            report["integrity"] = {stream: _integrity(plan_control(specs[stream], args.control),
                panels[stream], originals[(stream, specs[stream].candidate_id)], check) for stream in STREAMS}
            for left, right in zip(_cells(panels[STREAMS[0]]).values(), _cells(panels[STREAMS[1]]).values()):
                _same_cell(left, right)
            report["intervention_integrity_complete"] = True
            pair.update(outcome_contact=True, state="paired_development_contact_started")
            _record(handle, report)  # Durable paired contact BEFORE any target access.
            for stream in STREAMS:
                check()
                pair["streams"][stream] = {"state": "development_contact_started"}
                _record(handle, report)
                feedback = score_development_panel(args.scorer_directory, panels[stream].predictions,
                    panels[stream].calibration_means, check=check)
                available = all(value.available for value in feedback)
                original_regions = {value["region"]: value for value in
                    records[specs[stream].candidate_id][0]["streams"][stream]["regions"]}
                pair["streams"][stream] = {"regions": [asdict(value) for value in feedback],
                    "complete": available,
                    "global_selection_r2": sum(value.model_r2 for value in feedback) / 2 if available else None,
                    "change_from_finalist_r2": {value.region: (value.model_r2 - original_regions[value.region]["model_r2"]
                        if value.available else None) for value in feedback}}
                _record(handle, report)
            pair["state"] = "paired_development_scored"
            report["complete"] = all(pair["streams"][stream]["complete"] for stream in STREAMS)
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["ended_utc"] = utc()
            _record(handle, report)
    print(json.dumps({"mode": "score", "control": args.control, "complete": report["complete"],
                      "held_out_opened": False}), flush=True)
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("select", "fit", "score"))
    parser.add_argument("--last-index", type=int, choices=range(14))
    parser.add_argument("--control", type=int, choices=range(1, 11))
    parser.add_argument("--candidate-directory", type=Path)
    parser.add_argument("--scorer-directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname()
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) not in (1, 2)
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"):
        raise SystemExit("finalist controls require the selected1–2CPU/7GiB compute allocation")
    expected_arguments = {
        "select": (True, False, False, False), "fit": (False, True, True, False),
        "score": (False, True, False, True)}[args.mode]
    if tuple(value is not None for value in (args.last_index, args.control,
            args.candidate_directory, args.scorer_directory)) != expected_arguments:
        raise SystemExit("mode arguments cannot cross selection/candidate/scorer roles")
    episode = Path(__file__).resolve().parents[2]
    selected_out = episode / "outputs/runs/alternative_explanations/attempt001"
    if args.out.resolve() != selected_out or not args.out.is_dir():
        raise SystemExit("use the exact selected finalist-control directory")
    published = Path(os.environ["SCRATCH"]) / "br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002"
    for path, role in ((args.candidate_directory, "candidate"), (args.scorer_directory, "scorer")):
        if path is not None and path.resolve() != (published / role).resolve():
            raise SystemExit("only the exact published role directory is accepted")
    remaining = args.deadline_epoch - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= 3600:
        raise SystemExit("a future per-process deadline of at most one hour is required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    deadline = [args.deadline_epoch]
    def bind_deadline(original_expiry):
        deadline[0] = min(deadline[0], original_expiry)
    def check():
        if time.time() >= deadline[0]:
            raise TimeoutError("control-worker or original search-clock deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("finalist-control peak RSS exceeds6.5GiB")
    check()
    base = episode / "outputs/runs/matched_search"
    if args.mode == "select":
        return _select(args, base, bind_deadline, check)
    _, specs, progress, records = _selected(args, base, bind_deadline, check)
    _previous_controls(args.out, args.control, specs, check)
    return (_fit(args, specs, progress, records, check) if args.mode == "fit" else
            _score(args, specs, progress, records, check))


if __name__ == "__main__":
    sys.exit(main())
