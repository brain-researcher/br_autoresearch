"""Unused, explicit first-generation continuation of the original core reference.

The original initialized root and predecessor root/participant are read only.
A separately admitted invocation owns one fresh output directory. Only new
split records are published there; reused coefficients remain exact references
to original predecessor JSON/NPZ pairs. The coordinator must wait for the
predecessor to terminate. This caller neither polls nor schedules or retries.

The finalizer requires an explicit all30 participant-to-directory JSON mapping.
It accepts original completed people and completed first-generation resumes,
never searches for an available attempt, and preserves original G/T arithmetic.
Scores and model parameters stay in private files; stdout contains operational
status/counts only. No search credit, candidate comparison or scientific release
is granted. The unchanged RunBudget retains the original absolute deadline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import time

import numpy as np

import production_core_reference as core
import core_reference_split_resume as resume


CONTINUATION = "original_core_split_continuation_v1"
COUNTERS = ("reused_inner_splits_completed", "reused_outer_splits_completed",
            "new_inner_splits_completed", "new_outer_splits_completed",
            "inner_splits_completed", "outer_splits_completed",
            "actual_new_multivariate_fits", "original_logical_multivariate_fits")


def _original_settings(root, budget):
    root = core._path(root)
    settings = core._json(root / "settings.json", budget.record)
    complete = core._json(root / "inputs_complete.json", budget.record)
    recipe = core.validate_recipe(settings["recipe"])
    development, people = core._release(settings["release"], recipe["metadata_byte_cap"])
    core.require(settings["schema"] == core.SCHEMA
        and settings["recipe_id"] == core.digest(recipe)
        and settings["release_id"] == core.digest(settings["release"])
        and settings["development_id"] == core.digest(development)
        and tuple(settings["participants"]) == people
        and complete["status"] == "complete_core_reference_inputs"
        and complete["settings_id"] == core.digest(settings)
        and set(complete["participants"]) == set(people), "original_complete_role_bound_inputs_required")
    return root, settings, people


def _predecessor(root, person, settings_id, path, budget):
    predecessor = core._path(path)
    core.require(predecessor == root / person and predecessor.is_dir(),
                 "explicit_original_person_predecessor_required")
    saved = core._json(predecessor / "settings.json", budget.record)
    core.require(saved["participant_id"] == person and saved["settings_id"] == settings_id
        and saved["scientific_release_authorized"] is False and saved["audit_access"] is False
        and "continuation" not in saved and type(saved["cost_only"]) is bool,
        "original_predecessor_settings_required")
    return predecessor, saved


def _fresh_output(path, forbidden):
    output = core._path(path)
    resolved = output.resolve()
    core.require(all(resolved != value.resolve() and value.resolve() not in resolved.parents
                     for value in forbidden), "fresh_output_must_not_modify_source_directory")
    output.mkdir(mode=0o700, exist_ok=False)
    return output


def _reuse_binding(root, predecessor, progress):
    return dict(kind=CONTINUATION, source_root=str(root), predecessor_directory=str(predecessor),
        reused_inner_records=progress.get("reused_inner_records", []),
        reused_outer_records=progress.get("reused_outer_records", []),
        **{key: progress[key] for key in COUNTERS if key in progress})


def _validate_reuse(binding, root, predecessor, budget, array_cap):
    """Check bounded exact references, not scientific arrays or new provenance."""
    core.require(binding["kind"] == CONTINUATION and binding["source_root"] == str(root)
        and binding["predecessor_directory"] == str(predecessor)
        and all(type(binding[key]) is int and binding[key] >= 0 for key in COUNTERS),
        "complete_original_continuation_binding_required")
    inner, outer = set(), set()
    for kind, keys in (("inner", inner), ("outer", outer)):
        refs = binding["reused_" + kind + "_records"]
        core.require(isinstance(refs, list) and len(refs) <= (30 if kind == "inner" else 6),
                     "bounded_reused_split_references_required")
        for ref in refs:
            budget.check()
            i = ref["outer_index"]
            core.require(type(i) is int and 0 <= i < 6, "original_outer_index_required")
            if kind == "inner":
                j = ref["inner_index"]
                core.require(type(j) is int and 0 <= j < 6 and i != j, "original_inner_index_required")
                key, name = (i, j), "inner_%02d_%02d.json" % (i, j)
            else:
                key, name = i, "outer_%02d.json" % i
            core.require(key not in keys, "duplicate_reused_split_reference")
            keys.add(key)
            path = predecessor / name
            archive = path.with_suffix(".npz")
            meta = core._json(path, budget.record)
            core.require(ref["json_path"] == str(path) and ref["array_path"] == str(archive)
                and type(ref["array_bytes"]) is int and 0 < ref["array_bytes"] <= array_cap
                and meta["array_file"] == archive.name and meta["array_bytes"] == ref["array_bytes"]
                and archive.is_file() and archive.stat().st_size == ref["array_bytes"],
                "reused_original_split_reference_differs")
    core.require(all((i, j) in inner for i in outer for j in range(6) if i != j),
                 "reused_outer_requires_five_reused_inners")
    missing_pairs = sum((i, j) not in inner or (j, i) not in inner
                        for i in range(6) for j in range(i + 1, 6))
    core.require(binding["reused_inner_splits_completed"] == len(inner)
        and binding["reused_outer_splits_completed"] == len(outer)
        and binding["new_inner_splits_completed"] == 30 - len(inner)
        and binding["new_outer_splits_completed"] == 6 - len(outer)
        and binding["inner_splits_completed"] == 30 and binding["outer_splits_completed"] == 6
        and binding["original_logical_multivariate_fits"] == 438
        and binding["actual_new_multivariate_fits"] == 14 * missing_pairs + 3 * (6 - len(outer)),
        "complete_original_split_and_new_fit_counts_required")


def _public(report):
    result = {key: report[key] for key in
        ("status", "participant_id", "elapsed_seconds", "max_rss_kib", "error_class") if key in report}
    result.update({key: report.get("progress", {})[key] for key in COUNTERS
                   if key in report.get("progress", {})})
    return result


def run_participant(source_root, person, *, predecessor_directory, output_directory, budget):
    budget.require_compute_threads()
    inputs = core.bind_initialized(source_root, budget)
    core.require(person in inputs.people, "original_development_person_required")
    predecessor, saved = _predecessor(inputs.root, person, core.digest(inputs.settings),
                                      predecessor_directory, budget)
    core.require(not (predecessor / "complete.json").exists(), "completed_person_must_not_be_refitted")
    core.require(budget.durable - budget.written >= 65536, "operational_failure_reserve_required")
    directory = _fresh_output(output_directory, (predecessor,))
    budget.durable -= 65536
    started = time.monotonic()
    progress = dict(stage="loading_released_person", inner_splits_completed=0,
        outer_splits_completed=0, budget_checks=0, outer_index=None, inner_index=None)
    report = dict(status="incomplete_core_reference_person", participant_id=person,
        cost_only=saved["cost_only"], settings_id=core.digest(inputs.settings),
        scientific_release_authorized=False, audit_access=False,
        continuation=_reuse_binding(inputs.root, predecessor, progress))
    def check():
        progress["budget_checks"] += 1
        budget.check()
    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        inputs.check = check
        inputs.images.check = check
        core._publish_json(directory / "settings.json", report, budget)
        bound = inputs.load_participant(person)
        result = resume.fit_person(bound, inputs.recipe, predecessor_directory=predecessor,
            initialized_settings_id=core.digest(inputs.settings),
            metadata_byte_cap=min(inputs.recipe["metadata_byte_cap"], budget.record),
            array_byte_cap=inputs.recipe["npz_byte_cap"], check=check,
            retain_inner=lambda row: core._split_record(directory /
                ("inner_%02d_%02d.json" % (row.split.outer_index, row.split.inner_index)), row, budget),
            retain_outer=lambda row: core._save_outer(directory, row, budget), progress=progress)
        report["continuation"] = _reuse_binding(inputs.root, predecessor, progress)
        _validate_reuse(report["continuation"], inputs.root, predecessor, budget, inputs.recipe["npz_byte_cap"])
        core.require(result["participant_id"] == person, "resumed_person_identity_differs")
        core._publish_json(directory / "complete.json", {**report, **result,
            "status": "complete_core_reference_person", "inner_splits": 30, "outer_splits": 6,
            "models": core.CORE_MODELS, "concept_or_false_scores_computed": False, "search_credit": 0}, budget)
        report["status"] = "complete_core_reference_person"
    except Exception as error:
        report["error_class"] = type(error).__name__
        frames, frame = [], error.__traceback__
        while frame is not None and len(frames) < 24:
            frames.append(dict(file=Path(frame.tb_frame.f_code.co_filename).name,
                               function=frame.tb_frame.f_code.co_name, line=frame.tb_lineno))
            frame = frame.tb_next
        report["frames"] = frames
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(continuation=_reuse_binding(inputs.root, predecessor, progress), progress=progress,
            elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        core._operational_finish(directory, report, budget)
    return _public(report)


def finalize(source_root, participant_directories, output_directory, *, budget):
    """Aggregate explicit completed sources; arrays and model predictions stay closed."""
    budget.require_compute_threads()
    root, settings, people = _original_settings(source_root, budget)
    core.require(isinstance(participant_directories, dict) and set(participant_directories) == set(people),
                 "explicit_all30_original_role_directory_mapping_required")
    paths = {person: core._path(participant_directories[person]) for person in people}
    core.require(len({path.resolve() for path in paths.values()}) == 30, "distinct_person_sources_required")
    rows = []
    for person in people:
        budget.check()
        directory = paths[person]
        row = core._json(directory / "complete.json", budget.record)
        operation = core._json(directory / "operational.json", budget.record)
        core.require(row["status"] == operation["status"] == "complete_core_reference_person"
            and row["settings_id"] == operation["settings_id"] == core.digest(settings)
            and row["participant_id"] == person and row["inner_splits"] == 30 and row["outer_splits"] == 6
            and row["models"] == list(core.CORE_MODELS) and row["scientific_release_authorized"] is False
            and row["audit_access"] is False and row["concept_or_false_scores_computed"] is False
            and row["search_credit"] == 0, "all30_complete_identical_reference_people_required")
        if "continuation" in row:
            predecessor, _ = _predecessor(root, person, core.digest(settings), root / person, budget)
            core.require(directory.resolve() != predecessor.resolve()
                and predecessor.resolve() not in directory.resolve().parents
                and row["continuation"] == operation["continuation"], "distinct_completed_continuation_required")
            _validate_reuse(row["continuation"], root, predecessor, budget, settings["recipe"]["npz_byte_cap"])
        else:
            core.require(directory == root / person and "continuation" not in operation,
                         "original_completed_person_directory_required")
        core.require([item["outer_index"] for item in row["outer"]] == list(range(6)), "all_six_outer_pairs_required")
        for item in row["outer"]:
            core.require(type(item["image_index"]) is int and 0 <= item["image_index"] < 13
                and set(item["normalized_losses"]) == set(core.CORE_MODELS)
                and all(type(value) in (int, float) and np.isfinite(value) and value >= 0
                        for value in item["normalized_losses"].values()), "finite_complete_outer_losses_required")
        losses = row["normalized_participant_losses"]
        core.require(set(losses) == set(core.CORE_MODELS) and all(type(value) in (int, float)
            and np.isfinite(value) and value >= 0 for value in losses.values()), "finite_complete_reference_losses_required")
        for name in core.CORE_MODELS:
            core.require(losses[name] == float(np.mean([item["normalized_losses"][name] for item in row["outer"]])),
                         "retained_outer_and_person_mean_differ")
        rows.append(losses)
    gain = float(np.mean([row["nuisance"]-row["image"] for row in rows]))
    task = float(np.mean([row["reduced_task_nuisance"]-row["nuisance"] for row in rows]))
    record = dict(status="complete_30_person_core_calibration", schema=core.SCHEMA,
        settings_id=core.digest(settings), recipe_id=settings["recipe_id"], participants=people,
        participant_count=30, inner_splits=900, outer_splits=180,
        participant_normalized_losses=dict(zip(people, rows)), G=gain, T=task,
        delta_C=max(.001,.05*gain), delta_P=max(.0005,.025*gain),
        delta_Q=max(.002,.25*gain), delta_T=max(.001,.25*task),
        image_gain_positive=gain > 0, reference_underidentified=gain <= 0,
        search_credit=0, concept_or_false_scores_computed=False,
        scientific_release_authorized=False, audit_access=False,
        source_root=str(root), participant_directories={person:str(paths[person]) for person in people})
    # No owned final directory is created until every explicit source is complete.
    directory = _fresh_output(output_directory, tuple(paths.values()) + tuple(root / person for person in people))
    core._publish_json(directory / "core_reference.complete.json", record, budget)
    return dict(status=record["status"], participant_count=30, inner_splits=900, outer_splits=180)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("participant", "finalize"))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--participant-id")
    parser.add_argument("--predecessor-directory")
    parser.add_argument("--participant-directories-json")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, required=True)
    parser.add_argument("--durable-byte-limit", type=int, required=True)
    parser.add_argument("--record-byte-limit", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        budget = core.RunBudget(deadline_epoch=args.deadline_epoch, rss_limit_bytes=args.rss_limit_bytes,
            durable_byte_limit=args.durable_byte_limit, record_byte_limit=args.record_byte_limit, threads=args.threads)
        if args.phase == "participant":
            core.require(args.participant_id and args.predecessor_directory and args.participant_directories_json is None,
                         "explicit_first_generation_person_invocation_required")
            report = run_participant(args.source_root, args.participant_id,
                predecessor_directory=args.predecessor_directory, output_directory=args.output_directory, budget=budget)
        else:
            core.require(args.participant_id is None and args.predecessor_directory is None
                and args.participant_directories_json, "explicit_cohort_mapping_invocation_required")
            budget.require_compute_threads()
            mapping = core._json(core._path(args.participant_directories_json), budget.record)
            report = finalize(args.source_root, mapping, args.output_directory, budget=budget)
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        return 0 if not report["status"].startswith("incomplete") else 1
    except Exception as error:
        print(json.dumps(dict(status="core_reference_continuation_refused", error_class=type(error).__name__)), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
