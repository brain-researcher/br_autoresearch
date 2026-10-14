"""Inactive postfreeze selected-design pass, not a scientific release driver.

Consume the existing baseline producer's frozen plan, all30 completions and
six saved outer choices per person. Rebuild each original outer training set
and call the existing streamed all10-slot/all13-penalty diagnostic. No inner
search, concept/false fit or score, audit input or bank regeneration occurs.
The unchanged preparation fits/scores nuisance only to restore its original
training covariance. Its numerical and resource choices come from the freeze.

The participant action writes six complete diagnostic records and one owned
completion. The finalize action requires all30 literal participants and all180
folds with the same frozen choices and partition identities. Its result covers
selected-design matching ONLY: recovery, reliability, repeated-image and other
controls, precision, comparison and one-shot release remain unimplemented here.
No Boolean input can authorize those stages. A failed design cannot reopen
search or select another bank, penalty or participant.

This is an actual artifact/algorithm connector, but full-scale capacity cost is
unresolved. Nothing executes on import. The coordinator supplies an allocation,
native timeout, explicit RunBudget and a fresh output parent; no scheduler,
retry, source discovery or resource grant is implemented.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import continuous_fold_runner as folds
import continuous_tuning as tuning
import production_baseline_controller as baseline
from production_continuous_designs import bind_released_inputs, make_tuning_recipe, solver_limits
from selected_design_diagnostics import RECIPE_ID, FALSE_SLOTS, assemble_selected_outer_folds
from streamed_tuning_adapter import StreamedResources, outer_capacity_and_match
from training_transforms import CommonSensorSpace


SCOPE = "all30_selected_design_spectrum_joint_leverage_term_edf_only"


@dataclass(frozen=True)
class FrozenBaseline:
    root: Path
    settings: dict
    plan: dict
    fixed: dict
    trial: dict
    freeze_id: str
    people: tuple[str, ...]


def bind_frozen_baseline(search_dir, *, budget):
    """Read the existing producer schema; do not use its search-open helper."""
    require = baseline.require
    root = baseline._path(search_dir)
    settings = baseline._json(root / "settings.json", budget.record)
    plan = baseline.validate_plan(settings["plan"])
    release = baseline._json(baseline._path(plan["release_manifest"]), budget.record)
    development = baseline._json(baseline._path(release["development_manifest"]), budget.record)
    people = tuple(settings["participants"])
    require(development["role"] == "development"
            and people == tuple(development["participants"]) == tuple(sorted(people))
            and len(people) == len(set(people)) == 30
            and settings["plan_id"] == baseline.digest(plan)
            and settings["release_id"] == baseline.digest(release)
            and settings["development_role_id"] == baseline.digest(development)
            and settings["audit_access"] is False, "frozen_development_binding_differs")
    fixed = baseline._json(root / "baseline_fixed.json", budget.record)
    records = baseline._records(root, plan, budget)
    status = baseline.search_status(plan, records)
    core = baseline._core_record(root, plan, budget)
    trial = next((value for value in plan["trials"]
                  if value["trial_id"] == status["selected_trial_id"]), None)
    require(status["search_can_freeze"] and trial is not None
            and fixed["status"] == "baseline_fixed_precomparison_pending"
            and fixed["plan_id"] == settings["plan_id"]
            and fixed["search_status"] == status and fixed["selected_trial"] == trial
            and tuple(fixed["participants"]) == people
            and fixed["core_calibration_digest"] == baseline.digest(core)
            and fixed["meaningful_margins"] == core["meaningful_margins"]
            and fixed["positive_reference_gain"] == core["positive_reference_gain"]
            and fixed["scientific_release_authorized"] is False
            and fixed["adaptive_trials_after_this_freeze"] == 0
            and fixed["audit_opened"] is False
            and fixed["concept_or_false_scores_computed"] is False,
            "complete_original_baseline_freeze_required")
    # All predecessor participants must exist before any new neural payload is
    # opened, even when this invocation computes only one assigned participant.
    for person in people:
        budget.check()
        directory = root / (trial["trial_id"] + "__" + person)
        record = baseline._json(directory / "complete.json", budget.record)
        require(not (directory / "failure.json").exists()
                and record["status"] == "complete_participant_baseline"
                and record["plan_id"] == settings["plan_id"]
                and record["trial_id"] == trial["trial_id"]
                and record["recipe_id"] == baseline.digest(trial["recipe"])
                and record["participant_id"] == person
                and record["inner_splits"] == 30 and record["outer_splits"] == 6
                and record["concept_or_false_scores_computed"] is False
                and record["audit_access"] is False,
                "all30_complete_selected_baselines_required")
    return FrozenBaseline(root, settings, plan, fixed, trial, baseline.digest(fixed), people)


def saved_outer_selections(frozen, person, *, budget):
    """Use only scalar choices in existing JSON; do not decode score arrays."""
    baseline.require(person in frozen.people, "assigned_development_participant_required")
    directory = frozen.root / (frozen.trial["trial_id"] + "__" + person)
    result = []
    for outer in range(6):
        budget.check()
        path = directory / ("outer_%02d_result.json" % outer)
        saved = baseline._json(path, budget.record)
        record = saved["record"]
        arrays = path.with_suffix(".npz")
        baseline.require(saved["array_file"] == arrays.name and arrays.is_file()
                and type(saved["array_bytes"]) is int and saved["array_bytes"] == arrays.stat().st_size
                and record["split"] == baseline._plain(folds._split(outer)),
                "complete_original_outer_record_required")
        choice, match = record["choice"], record["match"]
        baseline.require(set(choice) == {"concept_index", "image_index"}
                and set(match) == {"concept_index", "label_index", "false_indices"},
                "original_outer_choice_schema_required")
        selected = tuning.BaselineChoice(tuning._index(choice["concept_index"]),
                                        tuning._index(choice["image_index"]))
        matched = tuning.MatchedIndices(tuning._index(match["concept_index"]),
            tuning._index(match["label_index"]), tuple(tuning._index(value) for value in match["false_indices"]))
        baseline.require(matched.concept_index == selected.concept_index
                and len(matched.false_indices) == 8, "original_outer_match_required")
        result.append((selected, matched))
    return tuple(result)


def partition_identities(release, person, *, budget):
    metadata = baseline._json(baseline._path(release["false_candidate_dirs"][person])
                              / "candidate_bank.json", budget.record)
    identities = tuple(metadata["primary_bank_fingerprints"])
    baseline.require(metadata["participant_id"] == person
            and len(identities) == len(set(identities)) == 8
            and all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                    for value in identities), "same_eight_literal_partition_identities_required")
    return identities


def selected_person(person, selections, identities, recipe, resources, *, check, retain):
    """Complete the original six outer checks; never invoke fixed_panel."""
    baseline.require(len(selections) == 6 and callable(check) and callable(retain),
                     "six_frozen_outer_selections_required")
    numerical_recipe = make_tuning_recipe(recipe)
    limits = solver_limits(recipe)
    sensor = CommonSensorSpace(person.inputs.channel_order)
    sequences = folds._roster(person.sequences, sensor, check)
    results = []
    for outer, (choice, original_match) in enumerate(selections):
        check()
        cache = tuning._prepare(sequences, folds._split(outer), sensor,
            numerical_recipe, person.prepare_training, limits, check)
        result = outer_capacity_and_match(cache, choice, numerical_recipe, check,
            resources=resources, diagnostic_recipe_id=RECIPE_ID,
            frozen_partition_fingerprints=identities)
        baseline.require(result.match == original_match,
                         "recomputed_outer_match_differs_no_penalty_fallback")
        retain(outer, result)
        # Persist full spectra, then retain only the small decision diagnostics;
        # do not accumulate six independent capacity-array payloads in memory.
        results.append(result.selected_design)
        del result, cache
    return assemble_selected_outer_folds(results)


def run_participant(search_dir, output_dir, participant_id, *, budget):
    budget.require_compute_threads()
    frozen = bind_frozen_baseline(search_dir, budget=budget)
    selections = saved_outer_selections(frozen, participant_id, budget=budget)
    parent = baseline._path(output_dir)
    baseline.require(parent.is_dir() and parent != frozen.root
            and not (parent / "cohort_complete.json").exists(), "fresh_selected_design_output_parent_required")
    directory = parent / participant_id
    directory.mkdir(mode=0o700, exist_ok=False)
    try:
        bound = bind_released_inputs(frozen.plan["release_manifest"], frozen.trial["recipe"],
                                     budget_check=budget.check)
        baseline.require(tuple(bound.people) == frozen.people, "released_cohort_changed")
        identities = partition_identities(bound.release, participant_id, budget=budget)
        person = bound.load_participant(participant_id)
        result = selected_person(person, selections, identities, frozen.trial["recipe"],
            StreamedResources(**frozen.plan["streamed_resources"]), check=budget.check,
            retain=lambda outer, value: baseline._split_record(
                directory / ("outer_%02d.json" % outer), value, budget))
        complete = {"status": "complete_participant_selected_design", "scope": SCOPE,
            "freeze_id": frozen.freeze_id, "plan_id": frozen.settings["plan_id"],
            "trial_id": frozen.trial["trial_id"], "recipe_id": baseline.digest(frozen.trial["recipe"]),
            "participant_id": participant_id, "frozen_selections": selections,
            "partition_fingerprints": identities, "result": result,
            "nuisance_preparation_performed": True, "concept_or_false_scores_computed": False,
            "full_false_bank_qualified": False, "scientific_release_authorized": False,
            "audit_access": False, "automatic_retry": False}
        baseline._publish_json(directory / "complete.json", complete, budget)
    except BaseException as error:
        try:
            baseline._publish_json(directory / "failure.json", {"exception_class": type(error).__name__,
                "complete": False, "scientific_release_authorized": False,
                "automatic_retry": False}, budget)
        except BaseException:
            pass
        raise
    return {"status": complete["status"], "participant_id": participant_id, "outer_folds": 6,
            "selected_design_passed": result["status"] == "selected_design_checks_pass",
            "scientific_release_authorized": False}


def _complete_record(record, frozen, person, selections, identities):
    """Check the concrete writer's sixfold schema; no external pass Boolean."""
    baseline.require(record["status"] == "complete_participant_selected_design" and record["scope"] == SCOPE
            and record["freeze_id"] == frozen.freeze_id and record["plan_id"] == frozen.settings["plan_id"]
            and record["trial_id"] == frozen.trial["trial_id"]
            and record["recipe_id"] == baseline.digest(frozen.trial["recipe"])
            and record["participant_id"] == person
            and record["frozen_selections"] == baseline._plain(selections)
            and tuple(record["partition_fingerprints"]) == identities
            and record["nuisance_preparation_performed"] is True
            and all(record[key] is False for key in ("concept_or_false_scores_computed",
                "full_false_bank_qualified", "scientific_release_authorized", "audit_access", "automatic_retry")),
            "complete_selected_design_binding_differs")
    result = record["result"]
    baseline.require(result["recipe_id"] == RECIPE_ID
            and result["scope"] == "selected_design_spectrum_joint_leverage_term_edf_only"
            and all(result[key] is False for key in ("false_bank_fully_qualified", "EEG_scored",
                                                   "bank_regeneration_or_penalty_fallback"))
            and len(result["outer_folds"]) == 6, "complete_sixfold_selected_result_required")
    passed = []
    for outer, value in enumerate(result["outer_folds"]):
        choice, matched = selections[outer]
        baseline.require(value["recipe_id"] == RECIPE_ID and value["outer_index"] == outer
                and tuple(value["frozen_partition_fingerprints"]) == identities
                and value["common_concept_penalty_index"] == choice.concept_index
                and tuple(item["false_slot"] for item in value["false_matches"]) == FALSE_SLOTS,
                "sixfold_identity_or_slot_coverage_differs")
        for item, index in zip(value["false_matches"], matched.false_indices):
            baseline.require(item["reference_penalty_index"] == choice.concept_index
                    and item["comparison_penalty_index"] == index
                    and type(item["term_edf_pass"]) is bool and type(item["passed"]) is bool,
                    "selected_false_penalty_or_result_differs")
            for metric in (item["spectrum"], item["joint_leverage"]):
                baseline.require(all(type(metric[key]) is bool for key in
                    ("standardized_pass", "relative_quantile_pass", "passed"))
                    and metric["passed"] == (metric["standardized_pass"] and metric["relative_quantile_pass"]),
                    "selected_distribution_completion_differs")
            baseline.require(item["passed"] == (item["term_edf_pass"] and item["spectrum"]["passed"]
                                                  and item["joint_leverage"]["passed"]),
                             "selected_false_match_completion_differs")
        all_passed = all(item["passed"] for item in value["false_matches"])
        baseline.require(type(value["passed"]) is bool and value["passed"] == all_passed,
                         "selected_fold_completion_differs")
        passed.append(all_passed)
    baseline.require(result["status"] == ("selected_design_checks_pass" if all(passed)
                     else "selected_design_match_failure"), "selected_person_disposition_differs")
    return all(passed)


def finalize(search_dir, output_dir, *, budget):
    budget.require_compute_threads()
    frozen = bind_frozen_baseline(search_dir, budget=budget)
    root = baseline._path(output_dir)
    baseline.require(root.is_dir() and root != frozen.root, "selected_design_output_parent_required")
    release = baseline._json(baseline._path(frozen.plan["release_manifest"]), budget.record)
    passed, records = [], {}
    for person in frozen.people:
        budget.check()
        directory = root / person
        record = baseline._json(directory / "complete.json", budget.record)
        baseline.require(not (directory / "failure.json").exists(), "partial_selected_design_cannot_complete")
        selections = saved_outer_selections(frozen, person, budget=budget)
        identities = partition_identities(release, person, budget=budget)
        passed.append(_complete_record(record, frozen, person, selections, identities))
        records[person] = baseline.digest(record)
        for outer in range(6):
            path = directory / ("outer_%02d.json" % outer)
            saved = baseline._json(path, budget.record)
            arrays = path.with_suffix(".npz")
            baseline.require(saved["array_file"] == arrays.name and arrays.is_file()
                    and arrays.stat().st_size == saved["array_bytes"]
                    and saved["record"]["selected_design"] == record["result"]["outer_folds"][outer],
                    "complete_saved_selected_design_fold_required")
    value = {"status": "all30_selected_design_pass" if all(passed) else "all30_selected_design_match_failure",
        "scope": SCOPE, "freeze_id": frozen.freeze_id, "plan_id": frozen.settings["plan_id"],
        "trial_id": frozen.trial["trial_id"], "participants": frozen.people,
        "participant_count": 30, "outer_folds": 180, "false_matches": 1440,
        "participant_record_digests": records, "all_selected_design_checks_passed": all(passed),
        "full_false_bank_qualified": False, "scientific_release_authorized": False,
        "concept_or_false_scores_computed": False, "audit_access": False,
        "required_next": "original_reliability_positive_recovery_precision_controls_and_fixed_comparison_caller",
        "bank_regeneration_or_search_reopening": False}
    baseline._publish_json(root / "cohort_complete.json", value, budget)
    return {key: value[key] for key in ("status", "participant_count", "outer_folds", "false_matches",
                                      "scientific_release_authorized")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("participant", "finalize"))
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participant-id")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, required=True)
    parser.add_argument("--durable-byte-limit", type=int, required=True)
    parser.add_argument("--record-byte-limit", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    budget = baseline.RunBudget(deadline_epoch=args.deadline_epoch, rss_limit_bytes=args.rss_limit_bytes,
        durable_byte_limit=args.durable_byte_limit, record_byte_limit=args.record_byte_limit, threads=args.threads)
    if args.action == "participant":
        result = run_participant(args.search_dir, args.output_dir, args.participant_id, budget=budget)
    else:
        baseline.require(args.participant_id is None, "finalize_cannot_select_participant")
        result = finalize(args.search_dir, args.output_dir, budget=budget)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
