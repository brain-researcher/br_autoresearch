"""Resume the original full EP18 core reference using bounded direct ridge.

One invocation completes one of the frozen 30 development participants.
Fifteen inner training sets yield the same thirty logical validations; six
outer folds use the original tuning order, image model and nuisance models.
Complete inner pairs and outer fits survive job-time limits. Public output
contains progress/cost only; scores remain private until all30 are complete.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import core_reference_paired_inner as paired
import production_baseline_controller as controller
import production_core_reference as core
from core_reference_direct_ridge import DirectCoreRidge
from training_transforms import CommonSensorSpace


class AllocatedBudget(controller.RunBudget):
    """Same guards with an actual finite job deadline under standing authority."""
    def __init__(self, *, deadline_epoch, rss_limit_bytes, durable_byte_limit,
                 record_byte_limit, threads):
        core.require(all(type(value) is int and value > 0 for value in
            (rss_limit_bytes, durable_byte_limit, record_byte_limit, threads)), "finite_positive_limits_required")
        start = int(os.environ["SLURM_JOB_START_TIME"])
        end = int(os.environ["SLURM_JOB_END_TIME"])
        core.require(np.isfinite(deadline_epoch) and start <= time.time() < deadline_epoch <= end-30,
                     "selected_actual_allocation_deadline_required")
        self.deadline, self.rss = float(deadline_epoch), rss_limit_bytes
        self.durable, self.record, self.threads = durable_byte_limit, record_byte_limit, threads
        self.written = 0


def atomic_json(path, record):
    data = json.dumps(controller._plain(record), sort_keys=True, allow_nan=False) + "\n"
    if len(data.encode()) > 16*1024**2:
        raise ValueError("Bounded JSON record required")
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _load_pair(path, first, second, binding):
    saved = controller._json(path, 16*1024**2)
    core.require(saved["binding"] == binding and saved["indices"] == [first, second],
                 "retained_pair_identity_changed")
    result = []
    for record, indices in zip(saved["results"], ((first, second), (second, first))):
        split = folds._split(*indices)
        core.require(record["split"] == controller._plain(asdict(split)), "retained_original_split_required")
        image, nuisance, normalized = [folds._readonly(record[name]) for name in
            ("image_sequence_losses", "nuisance_sequence_losses", "normalized_image_losses")]
        core.require(image.shape == (13, 12) and nuisance.shape == (12,) and normalized.shape == (13,)
            and np.isfinite(image).all() and np.isfinite(nuisance).all() and np.isfinite(normalized).all(),
            "complete_retained_inner_pair_required")
        result.append(controller.CoreInnerResult(split, tuple(record["training_sequence_ids"]),
            tuple(record["heldout_sequence_ids"]), image, nuisance, normalized))
    return tuple(result)


def compare_saved_scalars(path, cache, backend, limits, settings_id, budget):
    """Use already completed original LSMR fits; no new iterative reference job."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].tobytes())
        binding = metadata["binding"]
        core.require(binding["settings_id"] == settings_id and binding["participant_id"] == "sub-07"
            and binding["split"] == controller._plain(asdict(cache.split)) and binding["penalty_index"] == 0
            and metadata["design_shape"] == list(cache.train_designs["image"].shape),
            "saved_original_scalar_reference_identity_required")
        done = saved["completed"].copy()
        core.require(done.shape == (61,) and done.dtype == np.bool_ and done.any(),
                     "completed_original_reference_scalars_required")
        original = saved["coefficients"][:, done].copy()
        penalty = tuning._grid_penalty(cache, "image", 0)
        core.require(np.array_equal(saved["penalties"], penalty), "unchanged_reference_penalties_required")
    fitted = backend.fit(cache.train_designs["image"], cache.targets, cache.weights, penalty, limits, budget.check)
    beta = fitted.coefficients[:, done]
    gram, rhs = backend._statistics(cache.train_designs["image"], cache.targets, cache.weights)
    target = np.sum(cache.weights[:, None] * cache.targets[:, done]**2, axis=0)
    def objective(value):
        return target - 2*np.sum(value*rhs[:, done], axis=0) + np.sum(value*(gram @ value), axis=0) + np.sum(penalty[:, None]*value**2, axis=0)
    old, new = objective(original), objective(beta)
    core.require(np.isfinite(old).all() and np.isfinite(new).all() and np.all(old > 0)
        and np.all(new <= old + 1e-10*np.maximum(target, old)),
        "direct_solution_must_not_worsen_original_weighted_ridge_objective")
    difference = beta-original
    rms = np.sqrt(np.maximum(0., np.sum(difference*(gram @ difference), axis=0))/target)
    return {"original_completed_scalars": int(done.sum()), "same_penalty_index": 0,
        "maximum_relative_objective_increase": float(np.max((new-old)/old)),
        "maximum_prediction_difference_relative_to_weighted_target_RMS": float(np.max(rms)),
        "original_and_direct_weighted_objective_valid": True, "scientific_scores_released": False}


def run_person(initialized_root, run_root, public_root, person_id, budget, chunk_rows,
               reference_checkpoint=None):
    budget.require_compute_threads()
    inputs = core.bind_initialized(initialized_root, budget)
    core.require(person_id in inputs.people, "fixed_development_person_only")
    directory = Path(run_root) / person_id
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    public = Path(public_root) / (person_id + "_" + os.environ["SLURM_JOB_ID"])
    public.mkdir(mode=0o700, parents=True, exist_ok=False)
    binding = {"participant_id": person_id, "settings_id": controller.digest(inputs.settings),
        "recipe_id": controller.digest(inputs.recipe), "backend": "weighted_gram_cholesky_v1",
        "data_role": "development", "audit_access": False}
    if (directory / "settings.json").exists():
        core.require(controller._json(directory/"settings.json", budget.record) == binding,
                     "unchanged_continuation_binding_required")
    else:
        atomic_json(directory/"settings.json", binding)
    core.require(not (directory/"complete.json").exists(), "completed_person_must_not_rerun")
    report = {"status": "incomplete_core_reference_person", "participant_id": person_id,
        "job_id": os.environ["SLURM_JOB_ID"], "inner_pairs_completed": 0, "outer_splits_completed": 0,
        "reused_inner_pairs": 0, "reused_outer_splits": 0,
        "audit_access": False, "concept_or_false_scores_computed": False}
    started = time.monotonic()
    backend = DirectCoreRidge(check=budget.check, chunk_rows=chunk_rows)
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")
    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started,
                      solver=dict(backend.stats))
        atomic_json(public/"progress.json", report)
        budget.check()
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        person = inputs.load_participant(person_id)
        sensor = CommonSensorSpace(person.inputs.channel_order)
        limits = folds.SolverLimits(**inputs.recipe["solver"])
        convention = SimpleNamespace(normalization=inputs.recipe["normalization"])
        sequences = folds._roster(person.sequences, sensor, budget.check)
        inner_by_split, outer = {}, []
        with backend.selected():
            for first in range(6):
                for second in range(first+1, 6):
                    report.update(outer_index=first, inner_index=second)
                    path = directory/("inner_pair_%02d_%02d.json" % (first, second))
                    if path.exists():
                        results = _load_pair(path, first, second, binding)
                        report["reused_inner_pairs"] += 1
                    else:
                        cache = core._prepare_core(sequences, folds._split(first, second), sensor,
                            inputs.recipe, person.prepare_training, limits, budget.check, progress)
                        reverse = paired._counterpart(cache, sequences, sensor, convention, budget.check)
                        if reference_checkpoint is not None and first == 0 and second == 1:
                            progress("compare_existing_original_scalar_solutions")
                            comparison = compare_saved_scalars(reference_checkpoint, cache, backend, limits,
                                binding["settings_id"], budget)
                            atomic_json(public/"original_scalar_comparison.json", comparison)
                        progress("complete_original_13_penalty_grid")
                        results = paired._inner_pair(cache, reverse, convention, limits, budget.check)
                        atomic_json(path, {"binding": binding, "indices": [first, second], "results": results})
                        del cache, reverse
                        backend.clear()
                    for result in results:
                        inner_by_split[(result.split.outer_index, result.split.inner_index)] = result
                    report["inner_pairs_completed"] += 1
                    progress("inner_pair_complete")
            core.require(set(inner_by_split) == {(i,j) for i in range(6) for j in range(6) if i != j},
                         "all30_original_inner_validations_required")
            for index in range(6):
                report.update(outer_index=index, inner_index=None)
                path = directory/("outer_%02d_complete.json" % index)
                if path.exists():
                    saved = controller._json(path, budget.record)
                    core.require(saved["binding"] == binding and saved["outer_index"] == index,
                                 "retained_outer_identity_changed")
                    row = saved["result"]
                    report["reused_outer_splits"] += 1
                else:
                    inner = [inner_by_split[(index,j)] for j in range(6) if j != index]
                    mean = np.mean([value.normalized_image_losses for value in inner], axis=0)
                    choice = min(range(13), key=lambda j: (mean[j], -tuning.RIDGE_GRID[j], j))
                    cache = core._prepare_core(sequences, folds._split(index), sensor, inputs.recipe,
                        person.prepare_training, limits, budget.check, progress)
                    progress("selected_original_outer_image_and_task_models")
                    fitted = controller._core_outer(cache, choice, convention, sensor, limits, budget.check)
                    saved_dir = directory/("outer_%02d_job_%s" % (index, os.environ["SLURM_JOB_ID"]))
                    saved_dir.mkdir(mode=0o700, exist_ok=False)
                    core._save_outer(saved_dir, fitted, budget)
                    row = {"outer_index": index, "image_index": choice,
                           "normalized_losses": dict(fitted.result.normalized_losses)}
                    atomic_json(path, {"binding": binding, "outer_index": index,
                        "parameter_directory": str(saved_dir), "result": row})
                    del cache, fitted
                    backend.clear()
                outer.append(row)
                report["outer_splits_completed"] += 1
                progress("outer_fold_complete")
        completion = {**binding, "status": "complete_core_reference_person", "outer": outer,
            "normalized_participant_losses": {name: float(np.mean([row["normalized_losses"][name]
                for row in outer])) for name in core.CORE_MODELS}, "inner_splits": 30, "outer_splits": 6,
            "models": list(core.CORE_MODELS), "search_credit": 0,
            "concept_or_false_scores_computed": False, "scientific_release_authorized": False}
        atomic_json(directory/"complete.json", completion)
        report["status"] = completion["status"]
    except Exception as error:
        report.update(error_class=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        report.update(elapsed_seconds=time.monotonic()-started, solver=dict(backend.stats))
        atomic_json(public/"operational.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialized-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--rss-limit-bytes", required=True, type=int)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--reference-checkpoint")
    args = parser.parse_args()
    budget = AllocatedBudget(deadline_epoch=args.deadline_epoch, rss_limit_bytes=args.rss_limit_bytes,
        durable_byte_limit=8*1024**3, record_byte_limit=16*1024**2, threads=args.threads)
    report = run_person(args.initialized_root, args.run_root, args.public_root, args.participant_id,
                        budget, args.chunk_rows, args.reference_checkpoint)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
