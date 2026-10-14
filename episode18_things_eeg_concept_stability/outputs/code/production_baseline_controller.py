"""Inactive, finite EP18 baseline-search and cohort-persistence connector.

The explicit prospective plan supplies every scientific recipe and search
interpretation. No plan or numeric recipe is selected by importing/running
this source. The supported interpretation counts each registered branch
level, uses strict cohort-label-loss improvement and fixed-plan-order ties,
and treats a declared one-factor baseline challenge as a falsifier. These are
prospective implementation choices requiring coordinator selection, not claims
that SEARCH_POLICY uniquely defines those words. Twelve coverage trials are
needed under this interpretation; at least ten postcoverage trials follow.
The original minimum19/maximum44, per-level3,40% falsifier and patience10
conditions are all enforced. Only expanded_image candidates are ranked. The
fixed core rank64/cosine12 reference is a separate nonranked calibration.
The initial label-family requested ranks are64; effective ranks still follow
the existing min/dimension/numerical-rank rules. Plan exhaustion before the
stopping conditions is explicitly incomplete, never padded by duplicate trials.

initialize_search(plan, output_dir, ...) freezes an explicit finite plan.
run_participant(search_dir, trial_id, participant_id, ...) performs all30
inner and6 outer splits for ONE of the complete frozen30-person roster; root may
schedule different participants in separate bounded allocations. finish_trial
requires all30 complete records before any cohort objective is available.
freeze_baseline closes the search once its original stopping rules are met.
There is no scheduler, automatic continuation, retry, participant replacement,
or partial-cohort winner. An interrupted person's directory remains partial.

Only nuisance, reduced-task, image, linear-label and capacity-matched-label
models are fitted/scored. True/false training designs are used for exact EDF
only. Original 13-point image/label tuning,5 inner folds,6 outer folds, paired
sequence losses, original61D targets and training-only transforms are reused.
The separate fixed-panel runner is NEVER called, including after a freeze.

Outer coefficients, all fitted feature/label/RBF/sensor/whitening parameters,
penalties and solver diagnostics are saved without pickle. Input predictors
are reconstructed from the bound released artifacts, not copied into these
checkpoints. Baseline results contain no concept/false scores. A frozen result
is baseline_fixed_precomparison_pending, not scientific/audit authorization:
all30×6 selected-design checks, original recovery/reliability/precision gates,
fixed-panel/control evaluation and one-shot scientific release remain separate.

The CLI is a connector, not a resource grant. It requires the actual compute
allocation, explicit deadline/RSS/durable/record/thread limits, and uses only
explicit released paths. Native-call termination and the episode CPU/GPU
ledger remain the coordinator's responsibility. No work executes on import.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import time
from types import MappingProxyType

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
from feature_extractors import require_compute
from production_continuous_designs import (BASIS_WIDTHS, TrainingDesignBuilder,
    bind_released_inputs, make_tuning_recipe, solver_limits, validate_recipe)
from streamed_tuning_adapter import StreamedResources, capacity_for_split
from training_transforms import CommonSensorSpace


PLAN_SCHEMA = "ep18_explicit_baseline_search_v1"
SEMANTICS = {
    "coverage": "each_registered_level_at_least_three",
    "falsifier": "prespecified_single_factor_baseline_challenge",
    "objective": "mean_30_participant_mean_six_outer_matched_label_loss",
    "improvement": "strictly_lower_no_epsilon",
    "tie_break": "earlier_fixed_plan_position",
    "patience": "ten_consecutive_valid_nonimprovements_stop_only_after_all_minima",
    "terminal_panel": "expanded_image",
    "core_reference": "nonranked_core64_raised_cosine12",
    "label_ranks": "requested64_each_family",
}
CORE_ID = "core_reference"
CORE_MODELS = ("nuisance","reduced_task_nuisance","image")
FACTORS = ("temporal_basis", "image_rank", "label_kernel_bandwidth", "image_panel")
LEVELS = {"temporal_basis": tuple(BASIS_WIDTHS), "image_rank": (32,64,128),
          "label_kernel_bandwidth": (.5,1.,2.)}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
LAST_DEADLINE = 1791927000


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _path(value):
    result = Path(value)
    require(result.is_absolute() and ".." not in result.parts, "explicit_absolute_path_required")
    return result


def _identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value), "bounded_identifier_required")
    return value


def _json(path, cap):
    require(path.is_file() and path.stat().st_size <= cap, "bounded_metadata_file_required")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, (dict, MappingProxyType)):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class RunBudget:
    """Explicit owned-output and process bounds; no thread/allocation changes."""
    def __init__(self, *, deadline_epoch, rss_limit_bytes, durable_byte_limit, record_byte_limit, threads):
        require(isinstance(deadline_epoch, (int,float)) and not isinstance(deadline_epoch, bool)
                and math.isfinite(deadline_epoch) and time.time() < deadline_epoch <= LAST_DEADLINE,
                "finite_original_deadline_required")
        require(all(type(value) is int and value > 0 for value in
                    (rss_limit_bytes, durable_byte_limit, record_byte_limit, threads)), "explicit_positive_limits_required")
        self.deadline = float(deadline_epoch)
        self.rss, self.durable, self.record, self.threads = rss_limit_bytes, durable_byte_limit, record_byte_limit, threads
        self.written = 0

    def check(self):
        if time.time() >= self.deadline:
            raise TimeoutError("selected_deadline_reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.rss:
            raise MemoryError("selected_RSS_limit_reached")

    def require_compute_threads(self):
        require_compute()
        require(all(os.environ.get(name) == str(self.threads) for name in THREAD_ENV)
                and int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) >= self.threads,
                "actual_allocation_and_explicit_thread_caps_required")
        self.check()

    def admit_write(self, amount):
        self.check()
        if self.written + amount > self.durable:
            raise MemoryError("owned_output_byte_limit_reached")


def _publish_json(path, value, budget):
    data = (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    require(len(data) <= budget.record, "record_byte_limit_reached")
    budget.admit_write(len(data))
    temporary = path.with_name(path.name+".partial")
    with temporary.open("xb") as target:
        os.chmod(temporary, 0o600)
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    os.link(temporary, path)
    temporary.unlink()
    budget.written += len(data)


def _factor_values(recipe):
    ranks = set(recipe["image_ranks"].values())
    require(len(ranks) == 1, "explicit_common_image_rank_per_trial_required")
    return {"temporal_basis": recipe["temporal_basis"], "image_rank": int(next(iter(ranks))),
            "label_kernel_bandwidth": float(recipe["label_kernel_bandwidth"]), "image_panel": recipe["image_panel"]}


def _fixed_recipe(recipe):
    return {key: value for key, value in recipe.items() if key not in
            ("recipe_id", "temporal_basis", "image_ranks", "image_panel", "label_kernel_bandwidth")}


def validate_plan(plan):
    """Explicit finite plan; supported semantics are prospective, not defaults."""
    plan = json.loads(json.dumps(plan, allow_nan=False))
    require(set(plan) == {"schema", "semantics", "release_manifest", "streamed_resources", "core_calibration_recipe", "trials"}
            and plan["schema"] == PLAN_SCHEMA and plan["semantics"] == SEMANTICS,
            "explicit_prospective_search_semantics_required")
    _path(plan["release_manifest"])
    StreamedResources(**plan["streamed_resources"])
    core = validate_recipe(plan["core_calibration_recipe"])
    require(core["image_panel"] == "core_image" and core["temporal_basis"] == "raised_cosine_12"
            and set(core["image_ranks"].values()) == {64} and set(core["label_ranks"].values()) == {64},
            "fixed_nonranked_core_reference_required")
    trials = plan["trials"]
    require(isinstance(trials, list) and 19 <= len(trials) <= 44, "finite_original_trial_count_required")
    previous, seen_recipes, fixed = {}, set(), None
    for trial in trials:
        require(set(trial) == {"trial_id", "recipe", "kind", "falsifier_of", "changed_factor"}, "explicit_trial_fields_required")
        name = _identifier(trial["trial_id"])
        require(name != CORE_ID and name not in previous, "unique_trial_identity_required")
        recipe = validate_recipe(trial["recipe"])
        require(recipe["image_panel"] == "expanded_image" and set(recipe["label_ranks"].values()) == {64},
                "prospective_expanded_candidates_and_fixed_label_ranks_required")
        values = _factor_values(recipe)
        canonical = {key: value for key, value in recipe.items() if key != "recipe_id"}
        require(digest(canonical) not in seen_recipes, "duplicate_recipe_is_not_a_new_valid_trial")
        seen_recipes.add(digest(canonical))
        invariant = _fixed_recipe(recipe)
        require(fixed is None or invariant == fixed, "only_registered_search_factors_may_change")
        fixed = invariant
        require(_fixed_recipe(core) == fixed,"core_reference_keeps_the_same_nonsearch_numerical_recipe")
        require(trial["kind"] in ("ordinary", "falsifier"), "explicit_trial_role_required")
        if trial["kind"] == "falsifier":
            require(trial["falsifier_of"] in previous and trial["changed_factor"] in FACTORS,
                    "falsifier_requires_an_earlier_explicit_reference")
            reference = _factor_values(previous[trial["falsifier_of"]]["recipe"])
            changed = tuple(key for key in FACTORS if values[key] != reference[key])
            require(changed == (trial["changed_factor"],), "falsifier_must_be_exactly_one_registered_factor")
        else:
            require(trial["falsifier_of"] is None and trial["changed_factor"] is None, "ordinary_trial_has_no_falsifier_claim")
        previous[name] = trial
    # Check that the finite supplied plan can satisfy every minimum at all.
    coverage_records = [{"trial_id": trial["trial_id"], "objective": float(len(trials)-i),
                          "recipe_id": digest(trial["recipe"])} for i, trial in enumerate(trials)]
    status = search_status(plan, coverage_records, validate_records=False)
    require(status["minimum_requirements_met"], "supplied_plan_cannot_cover_original_search_requirements")
    return plan


def search_status(plan, records, *, validate_records=True):
    """Read completed cohort objectives only; no partial-person averages."""
    trials, records = plan["trials"], tuple(records)
    require(len(records) <= len(trials) <= 44, "completed_trial_count_differs")
    counts = {name: {str(level): 0 for level in levels} for name, levels in LEVELS.items()}
    coverage_at, best, patience, winner = None, None, 0, None
    post, challenges = 0, 0
    for index, (trial, record) in enumerate(zip(trials, records)):
        require(record["trial_id"] == trial["trial_id"] and record["recipe_id"] == digest(trial["recipe"])
                and type(record["objective"]) in (int,float) and math.isfinite(record["objective"])
                and record["objective"] >= 0, "complete_trial_identity_or_objective_differs")
        if validate_records:
            require(record["status"] == "complete_30_person_baseline_trial"
                    and record["plan_id"] == digest(plan) and record["audit_access"] is False
                    and record["participant_count"] == 30 and record["inner_splits"] == 900
                    and record["outer_splits"] == 180 and record["concept_or_false_scores_computed"] is False,
                    "only_complete_baseline_trials_count")
        values = _factor_values(trial["recipe"])
        for name in LEVELS:
            counts[name][str(values[name])] += 1
        if coverage_at is not None:
            post += 1
            challenges += trial["kind"] == "falsifier"
        if coverage_at is None and all(min(group.values()) >= 3 for group in counts.values()):
            coverage_at = index+1
        objective = record["objective"]
        if best is None or objective < best:
            best, winner, patience = objective, trial["trial_id"], 0
        else:
            patience += 1
    requirements = (len(records) >= 19 and coverage_at is not None and post >= 10
        and challenges*5 >= post*2 and winner is not None)
    exhausted = len(records) == len(trials)
    stopped = requirements and (patience >= 10 or len(records) == 44)
    return {"valid_trials": len(records), "coverage_counts": counts, "coverage_at": coverage_at,
            "postcoverage_trials": post, "postcoverage_falsifiers": challenges,
            "consecutive_nonimprovements": patience, "minimum_requirements_met": requirements,
            "search_can_freeze": stopped, "plan_exhausted": exhausted,
            "plan_incomplete": exhausted and not stopped,
            "selected_trial_id": winner if stopped else None}


def initialize_search(plan_file, output_dir, *, budget):
    budget.require_compute_threads()
    plan = validate_plan(_json(_path(plan_file), budget.record))
    release = _json(_path(plan["release_manifest"]), budget.record)
    development = _json(_path(release["development_manifest"]), budget.record)
    people = tuple(development["participants"])
    require(development["role"] == "development" and len(people) == len(set(people)) == 30
            and people == tuple(sorted(people)), "fixed_complete_development_role_required")
    output = _path(output_dir)
    output.mkdir(mode=0o700, exist_ok=False)
    _publish_json(output/"settings.json", {"plan": plan, "plan_id": digest(plan),
        "release_id": digest(release), "development_role_id":digest(development),
        "participants": people, "audit_access": False}, budget)
    return {"status": "baseline_search_initialized", "planned_trials": len(plan["trials"]), "participant_count": 30}


def _settings(search_dir, budget):
    root = _path(search_dir)
    value = _json(root/"settings.json", budget.record)
    plan = validate_plan(value["plan"])
    release = _json(_path(plan["release_manifest"]),budget.record)
    development = _json(_path(release["development_manifest"]),budget.record)
    require(value["plan_id"] == digest(plan)
            and value["release_id"] == digest(release)
            and value["development_role_id"] == digest(development)
            and value["participants"] == development["participants"],
            "frozen_plan_or_release_changed")
    require(not (root/"baseline_fixed.json").exists(), "baseline_already_fixed_no_more_search")
    return root, value, plan


def _records(root, plan, budget):
    records = []
    settings = _json(root/"settings.json",budget.record)
    for trial in plan["trials"]:
        path = root/(trial["trial_id"]+".complete.json")
        if not path.exists():
            break
        record = _json(path,budget.record)
        require(record["plan_id"] == settings["plan_id"]
                and record["participants"] == settings["participants"],"completed_cohort_binding_differs")
        records.append(record)
    return records


def _next_trial(root, plan, trial_id, budget):
    records = _records(root, plan, budget)
    calibration = root/(CORE_ID+".complete.json")
    if trial_id == CORE_ID:
        require(not calibration.exists() and not records,"core_reference_already_complete_or_candidates_started")
        return {"trial_id":CORE_ID,"recipe":plan["core_calibration_recipe"]}
    _core_record(root,plan,budget)
    status = search_status(plan, records)
    require(not status["search_can_freeze"] and len(records) < len(plan["trials"]), "search_is_ready_to_close")
    trial = plan["trials"][len(records)]
    require(trial["trial_id"] == trial_id, "trials_must_follow_the_fixed_prospective_order")
    return trial


def _core_record(root,plan,budget):
    core = _json(root/(CORE_ID+".complete.json"),budget.record)
    settings = _json(root/"settings.json",budget.record)
    require(core["status"] == "complete_30_person_core_calibration"
            and core["plan_id"] == digest(plan) and core["trial_id"] == CORE_ID
            and core["recipe_id"] == digest(plan["core_calibration_recipe"])
            and core["participants"] == settings["participants"]
            and core["participant_count"] == 30 and core["inner_splits"] == 900
            and core["outer_splits"] == 180 and core["audit_access"] is False
            and core["concept_or_false_scores_computed"] is False,
            "complete_nonranked_core_reference_required")
    return core


def _inner(cache, recipe, limits, resources, check):
    capacity = capacity_for_split(cache, recipe, check, resources=resources)
    feasible = tuning.all_feasible_tuples(capacity)
    matches = MappingProxyType({option.concept_index: tuning._match(recipe, cache.split, capacity, option)
                                for option in feasible})
    losses = {"nuisance": folds._readonly(cache.nuisance_losses[None,:])}
    for name in ("reduced_task_nuisance", "label_linear"):
        fit = folds._fit(cache.train_designs[name], cache.targets, cache.weights, cache.penalties[name], limits, check)
        losses[name] = folds._readonly(tuning._score(cache, name, fit, check)[None,:])
    for name in ("image", "label_capacity_matched"):
        values = []
        for index in range(13):
            fit = folds._fit(cache.train_designs[name], cache.targets, cache.weights,
                             tuning._grid_penalty(cache, name, index), limits, check)
            values.append(tuning._score(cache, name, fit, check))
        losses[name] = folds._readonly(values)
    normalized = {name: folds._readonly(tuning.normalize_losses(value, cache.nuisance_losses,
        normalization=recipe.normalization)) for name, value in losses.items()}
    return tuning.InnerGridResult(cache.split, tuple(item.sequence_id for item in cache.training),
        tuple(item.sequence_id for item in cache.heldout), MappingProxyType(losses),
        MappingProxyType(normalized), capacity, feasible, matches)


def _outer(cache, choice, recipe, sensor, limits, resources, check):
    capacity = capacity_for_split(cache, recipe, check, resources=resources)
    feasible = tuning.all_feasible_tuples(capacity)
    option = next((item for item in feasible if item.concept_index == choice.concept_index), None)
    require(option is not None, "underidentified_outer_capacity_match")
    match = tuning._match(recipe, cache.split, capacity, option)
    penalties = dict(cache.penalties)
    penalties["image"] = tuning._grid_penalty(cache, "image", choice.image_index)
    penalties["label_capacity_matched"] = tuning._grid_penalty(cache, "label_capacity_matched", match.label_index)
    require(set(penalties) == set(folds.BASELINE_MODELS), "baseline_only_fit_panel_required")
    fits, errors = {"nuisance": cache.nuisance_fit}, {"nuisance": cache.nuisance_losses}
    for name, penalty in penalties.items():
        if name != "nuisance":
            fits[name] = folds._fit(cache.train_designs[name], cache.targets, cache.weights, penalty, limits, check)
            errors[name] = tuning._score(cache, name, fits[name], check)
    error_record = folds.FoldErrors(cache.split, tuple(item.sequence_id for item in cache.training),
        tuple(item.sequence_id for item in cache.heldout), tuple(int(item.retained_mask.sum()) for item in cache.heldout),
        MappingProxyType(errors))
    normalized = MappingProxyType({name: float(tuning.normalize_losses(value, cache.nuisance_losses,
        normalization=recipe.normalization)) for name, value in errors.items()})
    result = tuning.OuterTuningResult(cache.split, choice, match, capacity, feasible, error_record, normalized)
    return tuning.FittedTunedFold(result, cache.preparation, sensor, cache.whitener,
        MappingProxyType(penalties), MappingProxyType(fits))


@dataclass(frozen=True)
class CoreInnerResult:
    split: object
    training_sequence_ids: tuple
    heldout_sequence_ids: tuple
    image_sequence_losses: np.ndarray
    nuisance_sequence_losses: np.ndarray
    normalized_image_losses: np.ndarray


@dataclass(frozen=True)
class CoreOuterResult:
    split: object
    image_index: int
    errors: object
    normalized_losses: object


@dataclass(frozen=True)
class FittedCoreFold:
    result: CoreOuterResult
    preparation: object
    sensor_space: object
    whitener: object
    penalties: object
    fits: object


def _core_inner(cache,recipe,limits,check):
    """Fixed core reference: all13 image penalties, no added-slot fit or EDF."""
    values = []
    for index in range(13):
        fit = folds._fit(cache.train_designs["image"],cache.targets,cache.weights,
                         tuning._grid_penalty(cache,"image",index),limits,check)
        values.append(tuning._score(cache,"image",fit,check))
    losses = folds._readonly(values)
    normalized = folds._readonly(tuning.normalize_losses(losses,cache.nuisance_losses,
        normalization=recipe.normalization))
    return CoreInnerResult(cache.split,tuple(item.sequence_id for item in cache.training),
        tuple(item.sequence_id for item in cache.heldout),losses,cache.nuisance_losses,normalized)


def _core_outer(cache,image_index,recipe,sensor,limits,check):
    penalties = {name:cache.penalties[name] for name in CORE_MODELS if name != "image"}
    penalties["image"] = tuning._grid_penalty(cache,"image",image_index)
    fits, errors = {"nuisance":cache.nuisance_fit}, {"nuisance":cache.nuisance_losses}
    for name in CORE_MODELS[1:]:
        fits[name] = folds._fit(cache.train_designs[name],cache.targets,cache.weights,penalties[name],limits,check)
        errors[name] = tuning._score(cache,name,fits[name],check)
    fold_errors = folds.FoldErrors(cache.split,tuple(item.sequence_id for item in cache.training),
        tuple(item.sequence_id for item in cache.heldout),
        tuple(int(item.retained_mask.sum()) for item in cache.heldout),MappingProxyType(errors))
    normalized = MappingProxyType({name:float(tuning.normalize_losses(value,cache.nuisance_losses,
        normalization=recipe.normalization)) for name,value in errors.items()})
    return FittedCoreFold(CoreOuterResult(cache.split,image_index,fold_errors,normalized),
        cache.preparation,sensor,cache.whitener,MappingProxyType(penalties),MappingProxyType(fits))


@dataclass(frozen=True)
class BaselineParticipantResult:
    participant_id: str
    phase: str
    inner_splits: int
    outer_splits: int
    outer_choices: tuple
    normalized_participant_losses: object


def baseline_participant(person, recipe, resources, *, check, retain_inner, retain_outer,
                         core_reference=False):
    """Concrete full72-sequence/36-split baseline path; no global monkeypatch."""
    tuning_recipe = make_tuning_recipe(recipe)
    limits = solver_limits(recipe)
    sensor = CommonSensorSpace(person.inputs.channel_order)
    sequences = folds._roster(person.sequences, sensor, check)
    inner_count, outer = 0, []
    for outer_index in range(6):
        inner = []
        for inner_index in range(6):
            if inner_index == outer_index:
                continue
            cache = tuning._prepare(sequences, folds._split(outer_index,inner_index), sensor,
                tuning_recipe, person.prepare_training, limits, check)
            result = (_core_inner(cache,tuning_recipe,limits,check) if core_reference else
                      _inner(cache, tuning_recipe, limits, resources, check))
            retain_inner(result)
            inner.append(result)
            inner_count += 1
            del cache
        if core_reference:
            image_mean = np.mean([item.normalized_image_losses for item in inner],axis=0)
            # The explicit production tie convention is loss, then larger
            # registered ridge, then index, exactly as make_tuning_recipe.
            choice = min(range(13),key=lambda index:(image_mean[index],-tuning.RIDGE_GRID[index],index))
        else:
            choice = tuning._selection(tuple(inner), outer_index, tuning_recipe)
        cache = tuning._prepare(sequences, folds._split(outer_index), sensor, tuning_recipe,
            person.prepare_training, limits, check)
        fitted = (_core_outer(cache,choice,tuning_recipe,sensor,limits,check) if core_reference else
                  _outer(cache, choice, tuning_recipe, sensor, limits, resources, check))
        retain_outer(fitted)
        outer.append({"split":fitted.result.split,"choice":choice,
                      "match":None if core_reference else fitted.result.match,
                      "normalized_losses":dict(fitted.result.normalized_losses)})
        del fitted, cache, inner, result
    losses = MappingProxyType({name: float(np.mean([value["normalized_losses"][name] for value in outer]))
                              for name in (CORE_MODELS if core_reference else folds.BASELINE_MODELS)})
    check()
    return BaselineParticipantResult(person.participant_id,"core_reference" if core_reference else "baseline",
                                     inner_count,len(outer),tuple(outer),losses)


def _publish_arrays(path,arrays,budget):
    reserve = sum(value.nbytes for value in arrays.values()) + 65536*(len(arrays)+1)
    budget.admit_write(reserve)
    temporary = path.with_name(path.name+".partial")
    with temporary.open("xb") as output:
        os.chmod(temporary,0o600)
        np.savez(output,**arrays)
        output.flush()
        os.fsync(output.fileno())
    actual = temporary.stat().st_size
    require(actual <= reserve,"array_archive_exceeds_admitted_bytes")
    os.link(temporary,path)
    temporary.unlink()
    budget.written += actual
    return actual


def _split_record(path,record,budget):
    """Persist actual baseline result fields, large spectra in numeric NPZ."""
    arrays = {}
    def encode(value):
        numerical_tuple = (isinstance(value,tuple) and len(value) >= 13
                           and all(type(item) in (int,float,bool) for item in value))
        if isinstance(value,np.ndarray) or numerical_tuple:
            array = np.asarray(value)
            require(array.dtype.kind in "biuf" and np.isfinite(array).all(),"finite_result_array_required")
            name = "array_%04d" % len(arrays)
            arrays[name] = array
            return {"array":name,"dtype":array.dtype.str,"shape":array.shape,"tuple":numerical_tuple}
        if is_dataclass(value):
            return {item.name:encode(getattr(value,item.name)) for item in fields(value)}
        if isinstance(value,(dict,MappingProxyType)):
            return {str(key):encode(item) for key,item in value.items()}
        if isinstance(value,(tuple,list)):
            return [encode(item) for item in value]
        return _plain(value)
    description = encode(record)
    size = _publish_arrays(path.with_suffix(".npz"),arrays,budget)
    _publish_json(path,{"record":description,"array_file":path.with_suffix(".npz").name,"array_bytes":size},budget)


def _checkpoint(directory, fitted, budget):
    """Exact named parameters; no pickle, EEG targets, model predictions or discovery."""
    builder = fitted.preparation.build_designs
    require(isinstance(builder, TrainingDesignBuilder), "actual_production_training_builder_required")
    arrays, description = {}, {}
    def component(prefix, value):
        require(is_dataclass(value), "known_fitted_dataclass_required")
        record = {}
        for item in fields(value):
            data = getattr(value,item.name)
            if isinstance(data,np.ndarray):
                key = prefix+"__"+item.name
                require(data.dtype.kind in "biuf" and np.isfinite(data).all(), "finite_fitted_array_required")
                arrays[key], record[item.name] = data, {"array": key, "shape": data.shape, "dtype": data.dtype.str}
            else:
                record[item.name] = _plain(data)
        description[prefix] = record
    panels = [("image",builder.image_transform)]
    if "label_linear" in fitted.fits:
        panels.append(("label",builder.label_transform))
    for panel_name, panel in panels:
        description[panel_name] = {"panel":panel.panel,"family_order":panel.family_order}
        for family, transform in zip(panel.family_order,panel.families):
            component(panel_name+"__"+family,transform)
    if "label_linear" in fitted.fits:
        component("label_kernel",builder.label_kernel)
    component("whitener",fitted.whitener)
    arrays["sensor_basis"] = fitted.sensor_space.basis
    for name, fit in fitted.fits.items():
        arrays[name+"__coefficients"] = fit.coefficients
        arrays[name+"__penalties"] = fitted.penalties[name]
        description["model__"+name] = {"stop_codes":fit.stop_codes,"iterations":fit.iterations}
    path = directory/("outer_%02d.npz" % fitted.result.split.outer_index)
    actual = _publish_arrays(path,arrays,budget)
    _publish_json(path.with_suffix(".json"), {"split":fitted.result.split,
        "training_sequence_ids":fitted.preparation.training_sequence_ids,
        "channel_order":fitted.sensor_space.channel_order,"parameters":description,
        "array_bytes":actual,"baseline_models":list(fitted.fits)},budget)
    _split_record(directory/("outer_%02d_result.json" % fitted.result.split.outer_index),fitted.result,budget)


def run_participant(search_dir, trial_id, participant_id, *, budget):
    budget.require_compute_threads()
    root, settings, plan = _settings(search_dir,budget)
    trial = _next_trial(root,plan,_identifier(trial_id),budget)
    require(participant_id in settings["participants"], "frozen_development_participant_required")
    directory = root/(trial_id+"__"+_identifier(participant_id))
    directory.mkdir(mode=0o700,exist_ok=False)
    _publish_json(directory/"settings.json", {"plan_id":settings["plan_id"],"trial_id":trial_id,
        "recipe_id":digest(trial["recipe"]),"participant_id":participant_id},budget)
    try:
        bound = bind_released_inputs(plan["release_manifest"],trial["recipe"],budget_check=budget.check)
        require(tuple(bound.people) == tuple(settings["participants"]), "released_cohort_changed")
        person = bound.load_participant(participant_id)
        result = baseline_participant(person,trial["recipe"],StreamedResources(**plan["streamed_resources"]),
            check=budget.check,retain_inner=lambda value:_split_record(directory/("inner_%02d_%02d.json" %
                (value.split.outer_index,value.split.inner_index)),value,budget),
            retain_outer=lambda value:_checkpoint(directory,value,budget),core_reference=trial_id == CORE_ID)
        _publish_json(directory/"baseline_results.json",result,budget)
        status = "complete_participant_core_calibration" if trial_id == CORE_ID else "complete_participant_baseline"
        _publish_json(directory/"complete.json", {"status":status,
            "plan_id":settings["plan_id"],"trial_id":trial_id,"recipe_id":digest(trial["recipe"]),
            "participant_id":participant_id,"inner_splits":30,"outer_splits":6,
            "participant_losses":result.normalized_participant_losses,
            "concept_or_false_scores_computed":False,"audit_access":False,
            "written_bytes_before_completion":budget.written},budget)
    except BaseException as error:
        # Preserve the original exception and every partial checkpoint. Failure
        # metadata is categorical and cannot turn the attempt into completion.
        try:
            _publish_json(directory/"failure.json", {"exception_class":type(error).__name__,
                "complete":False,"no_automatic_retry":True},budget)
        except BaseException:
            pass
        raise
    return {"status":status,"participant_id":participant_id,"inner_splits":30,"outer_splits":6}


def finish_trial(search_dir,trial_id,*,budget):
    budget.require_compute_threads()
    root,settings,plan = _settings(search_dir,budget)
    trial = _next_trial(root,plan,_identifier(trial_id),budget)
    core_reference = trial_id == CORE_ID
    expected_status = "complete_participant_core_calibration" if core_reference else "complete_participant_baseline"
    models = CORE_MODELS if core_reference else folds.BASELINE_MODELS
    records = []
    for person in settings["participants"]:
        budget.check()
        directory = root/(trial_id+"__"+person)
        record = _json(directory/"complete.json",budget.record)
        require(not (directory/"failure.json").exists() and record["status"] == expected_status
            and record["plan_id"] == settings["plan_id"] and record["trial_id"] == trial_id
            and record["recipe_id"] == digest(trial["recipe"]) and record["participant_id"] == person
            and record["inner_splits"] == 30 and record["outer_splits"] == 6
            and record["concept_or_false_scores_computed"] is False and record["audit_access"] is False
            and set(record["participant_losses"]) == set(models), "complete_exact_participant_required")
        records.append(record)
    losses = {name:[record["participant_losses"][name] for record in records] for name in models}
    require(all(np.isfinite(value).all() and np.all(np.asarray(value)>=0) for value in losses.values()), "finite_complete_cohort_losses_required")
    complete = {"status":"complete_30_person_core_calibration" if core_reference else "complete_30_person_baseline_trial",
        "plan_id":settings["plan_id"],
        "trial_id":trial_id,"recipe_id":digest(trial["recipe"]),"participant_count":30,
        "participants":settings["participants"],"inner_splits":900,"outer_splits":180,
        "participant_losses":losses,
        "concept_or_false_scores_computed":False,"audit_access":False}
    if core_reference:
        g = float(np.mean(np.asarray(losses["nuisance"])-losses["image"]))
        t = float(np.mean(np.asarray(losses["reduced_task_nuisance"])-losses["nuisance"]))
        complete["meaningful_margins"] = {"G":g,"T":t,"delta_C":max(.001,.05*g),
            "delta_P":max(.0005,.025*g),"delta_Q":max(.002,.25*g),"delta_T":max(.001,.25*t)}
        complete["positive_reference_gain"] = g > 0
        complete["calibration_disposition"] = "positive_reference_gain" if g > 0 else "underidentified_no_positive_reference_gain"
        complete["valid_trial_credit"] = 0
    else:
        complete["objective"] = float(np.mean(losses["label_capacity_matched"]))
    _publish_json(root/(trial_id+".complete.json"),complete,budget)
    status = search_status(plan,_records(root,plan,budget))
    status["core_calibration_complete"] = core_reference or (root/(CORE_ID+".complete.json")).exists()
    return status


def freeze_baseline(search_dir,*,budget):
    budget.require_compute_threads()
    root,settings,plan = _settings(search_dir,budget)
    core = _core_record(root,plan,budget)
    records = _records(root,plan,budget)
    status = search_status(plan,records)
    require(status["search_can_freeze"], "original_search_requirements_not_complete")
    selected = next(value for value in plan["trials"] if value["trial_id"] == status["selected_trial_id"])
    value = {"status":"baseline_fixed_precomparison_pending","plan_id":settings["plan_id"],
        "selected_trial":selected,"search_status":status,"participants":settings["participants"],
        "core_calibration":CORE_ID+".complete.json","core_calibration_digest":digest(core),
        "meaningful_margins":core["meaningful_margins"],"positive_reference_gain":core["positive_reference_gain"],
        "scientific_release_authorized":False,"adaptive_trials_after_this_freeze":0,
        "required_next":"all30_sixfold_selected_design_and_original_precomparison_qualification",
        "audit_opened":False,"concept_or_false_scores_computed":False}
    _publish_json(root/"baseline_fixed.json",value,budget)
    return {"status":value["status"],"selected_trial_id":selected["trial_id"],"valid_trials":len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("initialize","participant","finish_trial","freeze"))
    parser.add_argument("--search-dir",required=True)
    parser.add_argument("--plan")
    parser.add_argument("--trial-id")
    parser.add_argument("--participant-id")
    parser.add_argument("--deadline-epoch",type=float,required=True)
    parser.add_argument("--rss-limit-bytes",type=int,required=True)
    parser.add_argument("--durable-byte-limit",type=int,required=True)
    parser.add_argument("--record-byte-limit",type=int,required=True)
    parser.add_argument("--threads",type=int,required=True)
    args = parser.parse_args()
    budget = RunBudget(deadline_epoch=args.deadline_epoch,rss_limit_bytes=args.rss_limit_bytes,
        durable_byte_limit=args.durable_byte_limit,record_byte_limit=args.record_byte_limit,threads=args.threads)
    if args.action == "initialize":
        require(args.plan is not None,"explicit_plan_required")
        result = initialize_search(args.plan,args.search_dir,budget=budget)
    elif args.action == "participant":
        result = run_participant(args.search_dir,args.trial_id,args.participant_id,budget=budget)
    elif args.action == "finish_trial":
        status = finish_trial(args.search_dir,args.trial_id,budget=budget)
        result = {key:status[key] for key in ("valid_trials","search_can_freeze","plan_exhausted")}
    else:
        result = freeze_baseline(args.search_dir,budget=budget)
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
